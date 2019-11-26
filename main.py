from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle

resume = False
restore_file = "NeatCheckpoints/neat-checkpoint-31"

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):

        self.env = gym_super_mario_bros.make('SuperMarioBros-1-3-v3')

        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0] / 8)
        iny = int(ob.shape[1] / 8)
        done = False

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0
        imgarray = []

        while not done:
            # cv2.namedWindow("main", cv2.WINDOW_NORMAL)
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)


            actions = net.activate(imgarray)
            action = np.argmax(actions)

            ob, rew, done, info = self.env.step(action)

            xpos = info['x_pos']

            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
                fitness += 10
            else:
                counter += 1

            if counter > 250:
                done = True

        print(fitness)
        return fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(4))

pe = neat.ParallelEvaluator(4, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
'''
#Our stuff
env = gym_super_mario_bros.make('SuperMarioBros-1-3-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


done = True
for step in range(2):
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



    t = env.action_space.sample()
    print(t)
    state, reward, done, info = env.step(t)
    print(env.action_space)
    env.render()

env.close()
'''