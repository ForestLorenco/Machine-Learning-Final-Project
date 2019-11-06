from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    """
    Reward function is composed of 3 values
    1) v: the difference in agent x values between states
        does the agent move right
    2) c: the difference in the game clock between frames
    3) d: death penalty
    
    note, if mario is stuck on pipe, reward tends to be several 0's and -1, should cancel run after 3 or 4 0s in a row
    and give a reward of -10 
    """



    state, reward, done, info = env.step(env.action_space.sample())
    print(state.shape)
    env.render()

env.close()