import gymnasium as gym


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

def _env_creator(ctx = None, render_mode = 'rgb_array'):
    from supersuit.generic_wrappers import resize_v1

    from ray.rllib.env.wrappers.atari_wrappers import MaxAndSkipEnv
    return (
        resize_v1(  # resize to 64x64 and normalize images
           	MaxAndSkipEnv(
                CropBottomObservation(
                    gym.make("CarRacing-v2", render_mode = render_mode)
                ),
                skip=4
            )
            ,x_size=64, y_size=64
        )
    )

env = CropBottomObservation(gym.make("CarRacing-v2", render_mode = "rgb_array"))
env = _env_creator(render_mode="rgb_array")
env.reset()

import pygame

# 画面サイズの設定
screen_width, screen_height = 640, 640
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("DreamerV3 Visualization")

# 色の定義
BLACK = (0, 0, 0)

clock = pygame.time.Clock()
for _ in range(1000):
	action = env.action_space.sample()
	obs, reward, terminated, truncated, info  = env.step(action)
	# 84,96,3を描画
	surface = pygame.surfarray.make_surface(obs)
	surface = pygame.transform.rotate(surface, -90)
	screen.fill(BLACK)
	scaled_surface = pygame.transform.scale(surface, (screen_width, screen_height))
	screen.blit(scaled_surface, (0, 0))
	pygame.display.flip()
	clock.tick(30)
env.close()
pygame.quit()