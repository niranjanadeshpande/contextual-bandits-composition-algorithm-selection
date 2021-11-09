# contextual-bandits-composition-algorithm-selection

## subsection 1


This project contains the code required to demonstrate the impact of using contectual multi-armed bandits for online service composition algorithm selection.
The file "thresholded_master_frame.pkl" contains execution data for four popular service composition algorithms on 6,144 composition tasks created using the WS-DREAM dataset.
These algorithms are: Multi-Constrained Shortest Path (MCSP), Ant Colony System Optimization (ACS), Genetic Algorithm (GA) and Particle Swarm Optimization (PSO).

The file "bandits_concise.py" contains code to run contextual bandits with different exploration strategies. It can be executed as follows:
python bandits_concise.py 3 0.3
where the first argument (3) denotes the exploration strategy to be used and the second argument denotes the exploration parameter.
Following are the values for different contextual bandits strategies:
1001: Greedy
888: Greedy full-information
999: Greedy no learning
3: Upper Confidence Bound (UCB)
4: Epsilon-Greedy (If the epsilon parameter is set to 0, this is equivalent to greedy)

The last file, "plots.py" generates two figures:
(1) A scatterplot that compares the performance of greedy, greedy no learning and greedy full-information to demonstrate that greedy makes better selections
(2) A lineplot that shows the sum of time and memory as a function of the number of composition tasks handled for each contextual bandits strategy, the Single Best Solver (SBS) and the Virtual Best Solver (VBS).

The plots and results obtained can also be found in the results folder.
