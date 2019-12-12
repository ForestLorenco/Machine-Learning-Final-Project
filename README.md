# Machine-Learning-Final-Project
Code based off of Vedant-Gupta523 sonic GA trainer

## Motivation
Video games are a massive industry and as they continue to get better graphics, desire for better AI increases. The problem is that creating good AI is difficult to hardcode. The solution, use machine learning to create intelligent AI with neural networks.

## Implementation
Here we use the genetic algorithm NEAT (**N**euro**E**volution of **A**ugmenting **T**opologies). This algorithm is based off of the biological process of natural selection and evolution. 
A population of neural networks is created. The population of individual genomes is maintained. Each genome contains two sets of genes that describe how to build an artificial neural network:
<ul>
<li>Node genes → each specifies a single neuron.

<li>Connection genes → each specifies a single connection between neurons.
</ul>

To evolve a solution to a problem:
<ul>
<li>User provides a fitness function which computes a single real number indicating the quality of an individual genome.
<li>Algorithm progresses through a user-specified number of generations, with each generation being produced by reproduction and mutation of the most fit individuals of the previous generation.
</ul>

Here we chose to use **Super Mario Bros** as the game to optimize. We got the implementation of SMB in open ai gym from **Christian Kauton** (https://github.com/Kautenja/gym-super-mario-bros
). We used the right only movement function provided by the library which simplifies marios movement and also used a graphical downgrade of the game for training.
