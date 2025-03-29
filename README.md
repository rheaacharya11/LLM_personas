**Multi-Persona Oracles for Fair Classification**
*Senior Thesis Research*

*Important Files*


* no_regret.py: This has the no regret learning algorithm. You can run it from the command line, specifying a constraint set path, a csv with the individuals being compared, and an (optional) judge subset.
* testing.py: This is the file that tests for fairness and accuracy on a subset of constraints.
* prompts/binary.yaml: This file has the prompts used in my experiment for each prompt type.
* src/comparison_elicitation.py: This has the code used to generate and query the LLM for new comparisons.
* src/fixed_comparison.py: This has the code used to query the LLM on a fixed set of comparisons.
* The constraint_sets folder has the constraint sets used in my experiments.
* The experiments folder has the experiments I ran, as well as a config file that can be modified to add extra experiments.
* The fixed_comparisons folder has the code for generating and visualizing new fixed constraint sets.
