# contextual-bandits-composition-algorithm-selection

## Overview
These files contain the source code and dataset required to replicate the results for the ICSA 2022 submission titled "Towards Robust SOA Implementation Using Online Composition Algorithm Selection".
This submission demonstrates the impact of using contextual multi-armed bandits for online service composition algorithm selection.

## Pre-Requisites

The project source code is in python and has been executed using :

- python-3.6
- numpy-1.20.3
- pandas-1.3.2
- matplotlib-3.4.2
- seaborn-0.11.2
- sklearn-0.24.2
- random
- math
- sys
- time
- pickle

## Description



To begin, download and unzip the file named "contextual-bandits-composition-algorithm-selection-main.zip"
This foldercontains the following files:

- The file "thresholded_master_frame.pkl" contains execution data for four popular service composition algorithms on 6,144 composition tasks created using the WS-DREAM dataset.
These algorithms are: Multi-Constrained Shortest Path (MCSP), Ant Colony System Optimization (ACS), Genetic Algorithm (GA) and Particle Swarm Optimization (PSO).

- The file "bandits_concise.py" contains code to run contextual bandits with different exploration strategies. It can be executed as follows:
python bandits_concise.py 3 0.3 where the first argument (3) denotes the exploration strategy to be used and the second argument denotes the exploration parameter.
Following are the values for different contextual bandits strategies:
  -1001: Greedy
  -888: Greedy full-information
  -999: Greedy no learning
  -3: Upper Confidence Bound (UCB)
  -4: Epsilon-Greedy (If the epsilon parameter is set to 0, this is equivalent to greedy)

-The last file, "plots.py" generates two figures:
- Item 3
  - Sub Item 1
  - A scatterplot that compares the performance of greedy, greedy no learning and greedy full-information to demonstrate that greedy makes better selections
  - A lineplot that shows the sum of time and memory as a function of the number of composition tasks handled for each contextual bandits strategy, the Single Best Solver (SBS) and the Virtual Best Solver (VBS).

The plots and results in the paper can also be found in the results folder.
