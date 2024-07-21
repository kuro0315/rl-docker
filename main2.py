import logging
import os
import subprocess

import numpy as np
import tqdm

import ray

os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "300"  # 5分ごとに保存

logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("gymnasium").setLevel(logging.ERROR)

import datetime

dt_now = datetime.datetime.now()

from ray import tune
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.core.columns import Columns
from ray.rllib.env.wrappers.atari_wrappers import MaxAndSkipEnv
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
devices = tf.config.list_logical_devices("GPU")
env_name = "ImageCarRacing-v1"

def _env_creator(ctx = None, render_mode = 'rgb_array'):
    import gymnasium as gym
    from supersuit.generic_wrappers import resize_v1

    from ray.rllib.algorithms.dreamerv3.utils.env_runner import NormalizedImageEnv
    
    print(f"{env_name} is created.")
    
    class CropBottomObservation(gym.ObservationWrapper):
        def __init__(self, env, crop_height=12):
            super().__init__(env)
            self.crop_height = crop_height
            
            old_shape = env.observation_space.shape
            new_shape = (old_shape[0] - crop_height, old_shape[1], old_shape[2])
            
            self.observation_space = gym.spaces.Box(
                low=0, 
                high=255, 
                shape=new_shape, 
                dtype=env.observation_space.dtype
            )

        def observation(self, observation):
            # 画像の下部crop_height分のピクセルを除去
            return observation[:-self.crop_height, :, :]
    
    return NormalizedImageEnv(
        resize_v1(  # resize to 64x64 and normalize images
            CropBottomObservation(
                MaxAndSkipEnv(
                    gym.make("CarRacing-v2", render_mode = render_mode)
                    ,skip=4   
                )
            )
            ,x_size=64, y_size=64
        )
    )

import gymnasium as gym

gym.register(env_name,_env_creator)
tune.register_env(env_name, _env_creator)

num_cpus = 32
num_gpus = 1
ray.init(
    num_cpus=num_cpus,
    num_gpus=num_gpus,
)

config = (
    DreamerV3Config()
    .environment(
        env_name
    )
    .resources(
        num_learner_workers=0 if num_gpus == 1 else num_gpus,
        num_gpus_per_learner_worker=1 if num_gpus else 0,
        # For each (parallelized) env, we should provide a CPU. Lower this number
        # if you don't have enough CPUs.
        # num_cpus_for_local_worker=4 * (num_gpus or 1), # Tune利用時のみ効くみたい
    )
    .rollouts(
        # If we use >1 GPU and increase the batch size accordingly, we should also
        # increase the number of envs per worker.
        # num_envs_per_worker=4 * (num_gpus or 1),
        remote_worker_envs=False,
    )
    .reporting(
        metrics_num_episodes_for_smoothing=(num_gpus or 1),
        report_images_and_videos=False,
        report_dream_data=False,
        report_individual_batch_item_stats=False,
    )
    .training(
        model_size="M",# Sなら4.45it/s
        training_ratio=1024,
        # horizon_H=50,
        # batch_size_B=16 * (num_gpus or 1),
    )
)

algo = config.build()

LEARN = True
RESTORE = False and LEARN
if LEARN:
    tuner : tune.Tuner
    if not RESTORE:
        pass
    else:
        raise NotImplementedError("Please set RESTORE to False to run the training.")
        algo.restore("")
    save_path = "/workspace/results/" + dt_now.strftime("%Y%m%d%H%M%S") + "/"
    try:
        for i in tqdm.tqdm(range(100000)):
            res = algo.train()
            if i % 10000 == 0:
                algo.save(f"{save_path}checkpoint{i}")
    except KeyboardInterrupt:
        print("Training is interrupted.")
        algo.save(f"{save_path}checkpoint_latest")
    except Exception as e:
        print(f"An error occurred: {e}")
        algo.save(f"{save_path}checkpoint_latest")
else:
    raise NotImplementedError("Please set LEARN to True to run the training.")
    algo.restore("")


# Use the vector env API.
env = gym.vector.make(
    env_name, 
    num_envs=1, 
    asynchronous=False,
    render_mode = "rgb_array" # "rgb_array"#
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


import pygame

# Pygameの初期化
pygame.init()

# framesフォルダ内を空にする
import os
import shutil

shutil.rmtree("/workspace/frames", ignore_errors=True)
os.mkdir("/workspace/frames")

# 画面サイズの設定
screen_width, screen_height = 640, 640
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("DreamerV3 Visualization")

# 色の定義
BLACK = (0, 0, 0)

clock = pygame.time.Clock()
frame_number = 0
rewards = 0
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
    
    decoder = rl_module.decoder
    
    h = states["h"]
    z = states["z"]
    
    x = decoder(h, z)
    
    x = tf.reshape(x, shape=(1, 64, 64, 3))
    x = tf.squeeze(x)
    
    # # xの値を0~1に正規化
    # x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
    # xの値を0~1にクリップ
    x = tf.clip_by_value(x, 0, 1)
    
    # xを画像として表示
    x_numpy = x.numpy()
    
    # Plot the RGB image
    # plt.imshow(x_numpy)
    # plt.axis('off')
    # plt.pause(0.001)
    # plt.draw()
    x_numpy = (x.numpy() * 255).astype(np.uint8)
    surface = pygame.surfarray.make_surface(x_numpy)
    surface = pygame.transform.rotate(surface, -90)
    screen.fill(BLACK)
    scaled_surface = pygame.transform.scale(surface, (screen_width, screen_height))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    clock.tick(30)
    
    # フレームを画像として保存
    pygame.image.save(screen, f"/workspace/frames/frame_{frame_number}.jpeg")
    frame_number += 1

    # Perform a step in the env.
    obs, reward, terminated, truncated, info = env.step(actions)
    # Not at the beginning of the episode anymore.
    is_first = 0.0
    rewards += reward
    # print(f"reward: {reward}")

print("END")
print(f"Total rewards: {rewards}")
pygame.quit()


subprocess.run([
    "ffmpeg",
    "-framerate", "24", 
    "-i", "/workspace/frames/frame_%d.jpeg", 
    "-c:v", "libx264", 
    "-pix_fmt", "yuv420p", 
    "/workspace/frames/game_video.mp4"
])