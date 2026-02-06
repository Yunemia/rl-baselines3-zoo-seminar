import gymnasium as gym
import numpy as np
import cv2

class RemoveFireWrapper(gym.Wrapper):
  """
  Removes the Fire action from the action space.
  Was used after the Atari Wrapper.
  """
  def __init__(self, env):
    super().__init__(env)
    self.action_space = gym.spaces.Discrete(5)
    # Mapping: new actions (0-4) -> old action IDs
    self._mapping = [0, 2, 3, 4, 5]

  def step(self, action):
    # map new actions on old action IDs
    mapped_action = self._mapping[action]
    return self.env.step(mapped_action)


class CropLifeandScoreWrapper(gym.ObservationWrapper):
  """
  Blacks out the Life and Score display from the observation space.
  Was used before the Atari Wrapper.
  """
  def __init__(self, env, top_crop=20, life_rect=(30, 20, 30, 10)):
    super().__init__(env)
    self.top_crop = top_crop
    self.life_rect = life_rect

    h, w, c = env.observation_space.shape
    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(h, w, c),
        dtype=np.uint8
    )

  def observation(self, obs):
    # Cuts off the upper image area (Score display)
    cropped = obs[self.top_crop:, :, :]

    # Pads the cutoff with black
    h, w, c = obs.shape
    black_pad = np.zeros((self.top_crop, w, c), dtype=np.uint8)
    frame_padded = np.vstack((black_pad, cropped))

    # Places a black rectangle at the position of the life display
    x, y, rect_w, rect_h = self.life_rect
    cv2.rectangle(frame_padded, (x, y), (x + rect_w, y + rect_h), (0, 0, 0), -1)

    return frame_padded


class RewardMinMaxScalingWrapper(gym.RewardWrapper):
  """
  Scales the rewards via Min-Max Scaling.
  Needs clip_reward from the Atari Wrapper to be turned off.
  Was used after the Atari Wrapper.
  """
  def __init__(self, env, r_min=0, r_max=500):
    super().__init__(env)
    self.r_min = r_min
    self.r_max = r_max

  def reward(self, reward):
    reward = max(self.r_min, min(self.r_max, reward))
    return (reward - self.r_min) / (self.r_max - self.r_min)

class StepPenaltyWrapper(gym.RewardWrapper):
  """
  Adds a penalty to the reward.
  Was used after the Atari Wrapper.
  """
  def __init__(self, env, penalty=-0.1):
    super().__init__(env)
    self.penalty = penalty

  def reward(self, reward):
    return reward + self.penalty


class RewardCompletionAndColoringWrapper(gym.RewardWrapper):
  """
  Identifies and increases the reward of coloring into the destination color and round completion.
  Needs clip_reward from the Atari Wrapper to be turned off.
  Was used after the Atari Wrapper.
  """
  def __init__(self, env):
    super().__init__(env)
    self.last_reward = 0

  def reward(self, rew):

    if rew == 25: # Coloring a cube into the destination color.
      new_rew = 2
    elif rew == 100 and self.last_reward == 100: # Round Completion
      new_rew = 2
    else:
      new_rew = np.sign(rew) # Everything else clipped like the Atari Wrapper does
    self.last_reward = rew
    return new_rew


