# What Do You Want From Me? Adapting to the Uncertainty of Human Preferences

## Pre-requisites

This code needs Python 2.7.
Since it relies on Gambit 15, you need to [build Gambit from source](https://gambitproject.readthedocs.io/en/latest/build.html#)
as well as its [Python extension](https://gambitproject.readthedocs.io/en/latest/build.html#building-the-python-extension).

## Module Description

* `abm_analysis.py`: This script generates CSV files with simulation 
results (`simulate_and_store` produces files in the format of `*_experiment_results.csv`) and then 
performs statistical analysis and writes its results to `metrics.csv`.
Its `run_parallel_simulations` function is used in multiple
scripts for running IMPACT+ simulations.
The `metrics.csv` includes, per scenario:
  * Mean evacuation time.
  * Standard deviation of evacuation time.
  * Minimum evacuation time.
  * Maximum evacuation time.
  * [Post-Hoc pairwise Dunn's test](https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn.html), against the other scenarios.
* `abm_gamemodel.py`: This script contains the `generate_game_model` function, that uses
Gambit to return a game-theoretic model of the robot-survivor interaction. It is used by
the `abm_runner.py` script.
* `abm_runner.py`: This is the script called within the Netlogo IMPACT+ simulation,
for providing a decision for the robot interacting with a survivor. In each call to this script,
we generate an instance of `AutonomicManagerController` equipped with
a `TypeAnalyser` instance (see `analyser.py`) for decisions based on identity
inference.
* `abm_sensitivity.py`: This scripts generates the data and plots for the sensitivity
analysis of evacuation time as a function of fall length and help factor.

