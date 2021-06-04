# Task-Guided Inverse Reinforcement Learning Under Partial Information
​
## Dependencies
​
This manual has been tested on a clean Ubuntu 20.04 LTS installation.
​
- The packages has been tested on Python 3.8 with numpy, [gurobipy](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_using_pip_to_install_gr.html), [stormpy](https://moves-rwth.github.io/stormpy/) installed.
- For visualization matplotlib is needed, in addition to [tikzplot](https://github.com/nschloe/tikzplotlib)
​
## Install the package
To install the package, go to the directory MCE_IRL_POMDPs where it has been extracted and execute
```
python3 -m pip install -e .
```
​
## Reproducibility Instructions
​
We provide the command in order to reproduce the reulst in the paper. Note that the directory 
```
examples/all_domains/
```
contains all the POMDP model descriptions you want to reproduce the expected reward and computation time of [SolvePOMDP](https://www.erwinwalraven.nl/solvepomdp/) and [SARSOP](https://github.com/AdaCompNUS/sarsop)
​
### Table 1.: Comparison with existing approaches
In order to obtain the computation time and reward in all the domains by our approach you need to execute the file  `bench_scpforward.py`
```
python3 bench_scpforward.py
```
​
Comment and uncomment lines in this file according to the benchmark instance you want to reproduce,
​
### Influence of side information
​
Execute the following commands to learn a policy and plot the policy. The following command can be executed for every other envronemnts such as `avoid_example`,  `evade_example`
​
```cd examples/maze_example
python3 final_exp_maze.py
python3 plot_maze_result.py
```