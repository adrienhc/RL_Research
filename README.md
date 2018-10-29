# Adrien Hadj-Chaib  -  RL Research, Autonomous Bipedal Locomotion

This project is conducted under the supervision of Masaki Nakada PhD, in UCLA's Magix Lab - Computer Graphics and Vision Laboratory.

## Introduction 

The project uses [OpenAI's Gym](https://github.com/openai/gym)
The code implements a PPO (Proximal Policy Optimization) algorithm which trains a Dart human bipedal model to walk forward autonomously.  

Most of the code I use comes from [OpenAI's Reinforcement Learning Baseline](https://github.com/openai/baselines) for the PP0 Algorithm.

The repository itself is coming from [Wenhao Yu's SymmetryCurriculumLocomotion repository](https://github.com/VincentYu68/SymmetryCurriculumLocomotion/tree/master/dart-env), which implements [this research project](https://arxiv.org/pdf/1801.08093.pdf) conducted at the Georgia Institute of Technology.


The model is simulated using [Dart](http://dartsim.github.io/)'s physics based model, as well as [PyDart2](https://pydart2.readthedocs.io/en/latest/), which is apythin binding for Dart.
Feel free to consult OpenAI's gym, baselines, as well as Dart's and Pydart's repositories and their respective README for more informations.

## This Repository

### Directories

	videos - videos of the training and final locomotion learnt by the model, for different behavior learning targets.  

	tensorflow - contains the tensorboard version of graph used by the Mirror and Symmetry Loss, PPO algorithm
				 to view it: $ cd tensorflow
				 	     $ tensorboard --logdir=.
				 connect to http://localhost:6006/  on your web browser to view the interactive graph

## Contact 

hadjchaib.adrien@gmail.com