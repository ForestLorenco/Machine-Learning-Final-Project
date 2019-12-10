import pickle       # pip install cloudpickle
import gym_super_mario_bros
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import visualize    # pip install graphviz

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

resume = True #set this to true if loading from a checkpoint
restore_file = "neat-checkpoint-584" #Specify checkpoint name here
averages = []
best = []

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
        self.env = JoypadSpace(env, RIGHT_ONLY)

    def work(self):

        ob = self.env.reset()

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

            #print("Test",self.env.action_space)
            actions = net.activate(imgarray)
            action = np.argmax(actions)

            ob, rew, done, info = self.env.step(action)

            xpos = info['x_pos']

            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
            else:
                counter += 1

            if counter > 250:
                done = True
            if info['flag_get']:
                print("Finished")
                done = True

        print("Worker Fitness:{}".format(xpos))
        return int(xpos)


def eval_genomes(genome, config):
    worker = Worker(genome, config)
    return worker.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

if resume == True:
    print("Restored file from here")
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(6, eval_genomes)

winner = p.run(pe.evaluate,1)
print(stats.most_fit_genomes)
averages = stats.best_genomes(500)
file = open("stats.csv","w")
i = 1
file.write("Generations, Best Fitness\n")
for g in averages:
    file.write(str(i)+ ","+str(g.fitness))
    i+= 1
file.close()

#visualize.draw_net(config, winner, True)
#visualize.plot_stats(stats, ylog=False, view=True)
#visualize.plot_species(stats, view=True)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output)
#print(winner)
