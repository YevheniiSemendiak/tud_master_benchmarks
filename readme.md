:wave:

### The experiments results and analysis
To reproduce the analysis, provided in the [master thesis](https://github.com/YevheniiSemendiak/tud_master_text/blob/gh-pages/semendiak_thesis.pdf), please perform following steps:
1. Clone this repository.
2. Install the dependencies: `pip install seaborn pandas numpy matplotlib`.
3. Execute the following [notebooks](https://jupyter.org/):
    - *analyse_parameter_tuning* to check the results of parameter tuning for each meta-heuristic.
    - *analyze_first_bench* to analyze the results of main experiment set, created to verify the proposed concept applicability.
    - *analyse_second_bench* to analyze the influence of modified version [BRISEv2](https://github.com/dpukhkaiev/BRISE2) (will be published soon) configuration influence on the performance of created *online selection hyper-heuristic with parameter control in low-level heuristics*. 

### Code
Together with the experiment results, this repository, yet partially, contains a source code of the developed system.
Mostly it is represented by the created [search space](./core_entities/search_space.py) representation approach.

The examples of code usage may be found in the [corresponding](./examples) folder.
