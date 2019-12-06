from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle


def run(file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

    genome = pickle.load(open(file, 'rb'))
    env = gym_super_mario_bros.make('SuperMarioBros-1-3-v3')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    info = {'distance': 0}
    try:
        while info['distance'] != 3252:
            obs = env.reset()
            done = False
            i = 0
            old = 40
            while not done:
                ob = cv2.resize(ob, (inx, iny))
                ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
                ob = np.reshape(ob, (inx, iny))

                imgarray = np.ndarray.flatten(ob)
                
                output = net.activate(imgarray)
                action = np.argmax(actions)
                s, reward, done, info = env.step(action)
                obs = s
                i += 1
                if i % 50 == 0:
                    if old == info['distance']:
                        break

                    else:
                        old = info['distance']
            print("Distance: {}".format(info['distance']))
        env.close()
    except KeyboardInterrupt:
        env.close()
        exit()
if __name__ == "__main__":
    run('winner.pkl')