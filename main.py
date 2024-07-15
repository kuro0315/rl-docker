import numpy as np

from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import try_import_tf
from ray import train, tune
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
tf1, tf, tfv = try_import_tf()
devices = tf.config.list_logical_devices("GPU")
print(devices)
env_name = "ImageCarRacing-v1"

def _env_creator(ctx, render_mode = 'rgb_array'):
    import gymnasium as gym
    from supersuit.generic_wrappers import resize_v1
    from ray.rllib.algorithms.dreamerv3.utils.env_runner import NormalizedImageEnv

    return NormalizedImageEnv(
        resize_v1(  # resize to 64x64 and normalize images
            gym.make("CarRacing-v2", render_mode = render_mode), x_size=64, y_size=64
        )
    )

import gymnasium as gym
gym.register(env_name,_env_creator)
tune.register_env(env_name, _env_creator)

num_gpus = 1

config = (
    DreamerV3Config()
    .environment(env_name)
    .resources(
        num_gpus = 1 if num_gpus else 0,
    )
    .env_runners(
        num_envs_per_env_runner=8 * (num_gpus or 1), 
        remote_worker_envs=True
    )
    .learners(
        num_learners=0 if num_gpus == 1 else num_gpus,
        num_gpus_per_learner=1 if num_gpus else 0,
    )
    .reporting(
        metrics_num_episodes_for_smoothing=(num_gpus or 1),
        report_images_and_videos=False,
        report_dream_data=False,
        report_individual_batch_item_stats=False,
    )
    .training(
        model_size="XS",
        training_ratio=512,
    )
)

tuner = tune.Tuner(
    "DreamerV3",
    param_space=config,
    run_config=train.RunConfig(
        name="DreamerV3" + env_name,
        storage_path = "/workspace/results",
        stop={
            "env_runners/episode_return_mean": 150,
            "training_iteration": 10,
        },
        checkpoint_config=train.CheckpointConfig(
            checkpoint_at_end=True
        ),
    ),
)

best_results = tuner.fit().get_best_result()

algo = config.build()
print(best_results.path)
print(best_results.checkpoint)
algo.restore("/workspace/results/DreamerV3ImageCarRacing-v1/DreamerV3_ImageCarRacing-v1_0fced_00000_0_2024-07-14_12-10-29/checkpoint_000000")

# Use the vector env API.
env = gym.vector.make(
    env_name, 
    num_envs=1, 
    asynchronous=False,
    render_mode = "human"
)

terminated = truncated = False
# Reset the env.
obs, _ = env.reset()
# Every time, we start a new episode, we should set is_first to True for the upcoming
# action inference.
is_first = 1.0

# Extract the actual RLModule from the local (Dreamer) EnvRunner.
rl_module = algo.workers.local_worker().module
# Get initial states from RLModule (note that these are always B=1, so this matches
# our num_envs=1; if you are using a vector env >1, you would have to repeat the
# returned states `num_env` times to get the correct batch size):
states = rl_module.get_initial_state()


while not terminated and not truncated:
    # Use the RLModule for action computations directly.
    # DreamerV3 expects this particular batch format: obs, prev. states and the
    # `is_first` flag.
    batch = {
        # states is already batched (B=1)
        Columns.STATE_IN: states,
        # obs is already batched (due to vector env).
        Columns.OBS: tf.convert_to_tensor(obs),
        # set to True at beginning of episode.
        "is_first": tf.convert_to_tensor([is_first]),
    }
    outs = rl_module.forward_inference(batch)
    # Extract actions (which are in one hot format) and state-outs from outs
    # actions = np.argmax(outs[Columns.ACTIONS].numpy(), axis=-1)
    actions = outs[Columns.ACTIONS].numpy()
    states = outs[Columns.STATE_OUT]

    # Perform a step in the env.
    obs, reward, terminated, truncated, info = env.step(actions)
    # Not at the beginning of the episode anymore.
    is_first = 0.0