This directory contains Python code to reproduce experimental results from our KDD 2025 paper [Benchmarking Fraud Detectors on Private Graph Data](https://dl.acm.org/doi/10.1145/3690624.3709170), by Alexander Goldberg, Giulia Fanti, Nihar Shah, and Zhiwei Steven Wu. 

# Setup 

We use conda to install necessary requirements. To recreate the conda environment used to run this code, execute:

    > conda env create -f environment.yml
    
which should create a conda environment called benchmark.

Then run:

    > conda activate benchmark
    
Once the environment is activated, you can excecute the provided python scripts to replicate experiments. You will want to create a directory named results/ under the root directory where experiment results are automatically saved after running experiments and loaded from for analysis.

We describe the main files below.

# Main Files

- `run_partition_agg_experiment.py`:  script to run experiments using the "Partition-Duplicate-Aggregate" algorithm on all datasets. Output saved to results/ directory.
- `analyze_partition_agg_experiment.py`: script to be run after executing run_partition_agg_experiment.py to analyze accuracy of using Partition-Duplicate-Aggregate at different privacy budgets.
- `run_synthetic_graph_experiment`: script to be run after executing run_partition_agg_experiment.py to analyze accuracy of using Partition-Duplicate-Aggregate at different privacy budgets.
- `analyze_synthetic_graph_experiment`: script to be run after executing run_synthetic_graph_experiment to analyze accuracy of using synthetic graph generation methods at different privacy budgets.
- `fraud_detectory.py`: Implementations of the fraud detectors benchmarked.
- `visualizations.ipynb`: Notebook to generate the charts used in the paper.
- `synthetic_algos/': Directory containing all synthetic graph data generation algorithm implementations
- `datasets`: Directory containing all code needed to load the graph data used.

