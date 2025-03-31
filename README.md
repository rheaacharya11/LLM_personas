**Multi-Persona Oracles for Fair Classification**
*Senior Thesis*

Here are some important files:
* no_regret.py: No Regret Learning Algorithm. To run it, you can do so in the command line, adding a path to the constraints set, the data of the individuals being compared, and (optionally) a subset of judges.
* src/fixed_comparison.py: Query the LLaMa model using the fixed comparisons.
* src/comparison_elicitation.py: Generate random comparisons from a larger constraint set, such that each pair is seen by at least 10 judges, and then query the LLaMa model.
* experiments/ This has config files that can be modified to add more experiments, as well as files to send the experiments to the cluster.
* fixed_comparison/ This has code to generate and visualize new sets of comparison pairs.
* no_regret_setup/ Some set up code for the no_regret experiments
