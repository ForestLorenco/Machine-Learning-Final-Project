from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle


def run(file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

    genome = pickle.load(open(file, 'rb'))
    #print(genome)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
    env = JoypadSpace(env, RIGHT_ONLY)

    env1 = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
    env1 = JoypadSpace(env1, RIGHT_ONLY)


    net = neat.nn.FeedForwardNetwork.create(genome, config)
    try:
        obs = env.reset()
        env1.reset()

        inx = int(obs.shape[0] / 8)
        iny = int(obs.shape[1] / 8)
        done = False
        while not done:
            env.render()
            env1.render()
            obs = cv2.resize(obs, (inx, iny))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (inx, iny))

            imgarray = np.ndarray.flatten(obs)

            actions = net.activate(imgarray)
            action =  np.argmax(actions)
            
            _,_,_,info1 = env1.step(action)
            s, reward, done, info = env.step(action)
            xpos = info['x_pos']


            print(done, action, xpos)
            obs = s
        env1.close()
        env.close()
    except KeyboardInterrupt:
        env.close()
        env1.close()
        exit()
if __name__ == "__main__":
    run('winner.pkl')