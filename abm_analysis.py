import multiprocessing
import time
from multiprocessing import Pool

import numpy as np
import pyNetLogo
import pandas as pd
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.api as sm

MODEL_FILE = "/home/cgc87/github/robot-assisted-evacuation/impact2.10.7/v2.11.0.nlogo"  # type:str
NETLOGO_HOME = "/home/cgc87/netlogo-5.3.1-64"  # type:str
NETLOGO_VERSION = "5"  # type:str

TURTLE_PRESENT_REPORTER = "count turtles"  # type:str
EVACUATED_REPORTER = "number_passengers - count agents + 1"  # type:str
DEAD_REPORTER = "count agents with [ st_dead = 1 ]"  # type:str

ENABLE_STAFF_COMMAND = "set REQUEST_STAFF_SUPPORT TRUE"  # type:str
ENABLE_PASSENGER_COMMAND = "set REQUEST_BYSTANDER_SUPPORT TRUE"

RESULTS_CSV_FILE = "data/experiment_results_nosupport_support.csv"  # type:str
NO_SUPPORT_COLUMN = "no-support"  # type:str
ONLY_STAFF_SUPPORT_COLUMN = "staff-support"  # type:str
ONLY_PASSENGER_SUPPORT_COLUMN = "passenger-support"  # type:str

SAMPLES = 5  # type:int


# Using https://www.stat.ubc.ca/~rollin/stats/ssize/n2.html
# And https://www.statology.org/pooled-standard-deviation-calculator/


# function to calculate Cohen's d for independent samples
# Inspired by: https://machinelearningmastery.com/effect-size-measures-in-python/

def cohen_d_from_metrics(mean_1, mean_2, std_dev_1, std_dev_2):
    # type: (float, float, float, float) -> float
    pooled_std_dev = np.sqrt((std_dev_1 ** 2 + std_dev_2 ** 2) / 2)
    return (mean_1 - mean_2) / pooled_std_dev


def calculate_sample_size(mean_1, mean_2, std_dev_1, std_dev_2, alpha=0.05, power=0.8):
    # type: (float, float, float, float, float, float) -> float
    analysis = sm.stats.TTestIndPower()  # type: sm.stats.TTestIndPower
    effect_size = cohen_d_from_metrics(mean_1, mean_2, std_dev_1, std_dev_2)
    result = analysis.solve_power(effect_size=effect_size,
                                  alpha=alpha,
                                  power=power,
                                  alternative="two-sided")
    return result


def run_simulation(run_id, post_setup_command):
    # type: (int, str) -> float
    netlogo_link.command("setup")
    if post_setup_command:
        netlogo_link.command(post_setup_command)
        print("{} {} executed".format(run_id, post_setup_command))
    else:
        print("{} no post-setup command".format(run_id))

    metrics_dataframe = netlogo_link.repeat_report(
        netlogo_reporter=[TURTLE_PRESENT_REPORTER, EVACUATED_REPORTER, DEAD_REPORTER], reps=2000)  # type: pd.DataFrame

    evacuation_finished = metrics_dataframe[
        metrics_dataframe[TURTLE_PRESENT_REPORTER] == metrics_dataframe[DEAD_REPORTER]]

    evacuation_time = evacuation_finished.index.min()  # type: float
    print("{} evacuation time {}".format(run_id, evacuation_time))

    return evacuation_time


def initialize(gui):
    # type: (bool) -> None
    global netlogo_link

    netlogo_link = pyNetLogo.NetLogoLink(netlogo_home=NETLOGO_HOME,
                                         netlogo_version=NETLOGO_VERSION,
                                         gui=gui)  # type: pyNetLogo.NetLogoLink
    netlogo_link.load_model(MODEL_FILE)


def start_experiments():
    start_time = time.time()  # type: float
    no_support_times = run_parallel_simulations(SAMPLES, post_setup_command="")  # type:List[float]
    support_times = run_parallel_simulations(SAMPLES, post_setup_command=ENABLE_PASSENGER_COMMAND)  # type:List[float]

    experiment_results = pd.DataFrame(data=list(zip(no_support_times, support_times)),
                                      columns=[NO_SUPPORT_COLUMN, ONLY_PASSENGER_SUPPORT_COLUMN])  # type:pd.DataFrame
    end_time = time.time()  # type: float
    print("Simulation finished after {} seconds".format(end_time - start_time))

    experiment_results.to_csv(RESULTS_CSV_FILE)
    print("Data written to {}".format(RESULTS_CSV_FILE))


def run_simulation_with_dict(dict_parameters):
    # type: (Dict) -> float
    return run_simulation(**dict_parameters)


def run_parallel_simulations(samples, post_setup_command, gui=False):
    # type: (int, str, bool) -> List[float]

    initialise_arguments = (gui,)  # type: Tuple
    simulation_parameters = [{"run_id": run_id, "post_setup_command": post_setup_command}
                             for run_id in range(samples)]  # type: List[Dict]

    results = []  # type: List[float]
    executor = Pool(initializer=initialize,
                    initargs=initialise_arguments)  # type: multiprocessing.pool.Pool

    for simulation_output in executor.map(func=run_simulation_with_dict,
                                          iterable=simulation_parameters):
        results.append(simulation_output)

    executor.close()
    executor.join()

    return results


def plot_results(results_dataframe):
    # type: (pd.DataFrame) -> None

    _ = sns.violinplot(data=results_dataframe)
    plt.show()
    _ = sns.stripplot(data=results_dataframe, jitter=True)
    plt.show()


def analyse_results():
    results_dataframe = pd.read_csv(RESULTS_CSV_FILE, index_col=[0])  # type: pd.DataFrame
    results_dataframe = results_dataframe.dropna()

    plot_results(results_dataframe)

    evacuation_no_support = results_dataframe[NO_SUPPORT_COLUMN].values  # type: List[float]
    mean_no_support = np.mean(evacuation_no_support).item()  # type:float
    std_dev_no_support = np.std(evacuation_no_support).item()  # type:float

    evacuation_with_support = results_dataframe[ONLY_PASSENGER_SUPPORT_COLUMN].values  # type: List[float]
    mean_staff_support = np.mean(evacuation_with_support).item()  # type:float
    std_dev_staff_support = np.std(evacuation_with_support).item()  # type:float

    print(
        "np.mean(evacuation_no_support) = {} np.std(evacuation_no_support) = {}"
        " len(evacuation_no_support)={}".format(mean_no_support, std_dev_no_support, len(evacuation_no_support)))
    print(
        "np.mean(evacuation_with_support) = {} np.std(evacuation_with_support) = {} "
        "len(evacuation_with_support) = {}".format(mean_staff_support, std_dev_staff_support,
                                                   len(evacuation_with_support)))
    print("Sample size: {}".format(
        calculate_sample_size(mean_no_support, mean_staff_support, std_dev_no_support, std_dev_staff_support)))

    null_hypothesis = "The distribution of {} times is THE SAME as the distribution of {} times".format(
        NO_SUPPORT_COLUMN, ONLY_PASSENGER_SUPPORT_COLUMN)  # type: str

    threshold = 0.05  # type:float
    u, p_value = mannwhitneyu(x=evacuation_no_support, y=evacuation_with_support)
    print("U={} , p={}".format(u, p_value))
    if p_value > threshold:
        print("FAILS TO REJECT: {}".format(null_hypothesis))
    else:
        print("REJECT: {}".format(null_hypothesis))


if __name__ == "__main__":
    start_experiments()

    plt.style.use('seaborn-darkgrid')
    analyse_results()
