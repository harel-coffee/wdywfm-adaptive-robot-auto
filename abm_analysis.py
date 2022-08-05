import numpy as np
import pyNetLogo
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.api as sm

TURTLE_PRESENT_REPORTER = "count turtles"  # type:str
EVACUATED_REPORTER = "number_passengers - count agents + 1"  # type:str
DEAD_REPORTER = "count agents with [ st_dead = 1 ]"  # type:str

ENABLE_STAFF_COMMAND = "set REQUEST_STAFF_SUPPORT TRUE"  # type:str
ENABLE_PASSENGER_COMMAND = "set REQUEST_BYSTANDER_SUPPORT TRUE"

RESULTS_CSV_FILE = "data/experiment_results_nosupport_support.csv"  # type:str
NO_SUPPORT_COLUMN = "no-support"  # type:str
ONLY_STAFF_SUPPORT_COLUMN = "staff-support"  # type:str
ONLY_PASSENGER_SUPPORT_COLUMN = "passenger-support"  # type:str

SAMPLES = 400  # type:int


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


def run_simulation(netlogo_link, post_setup_command=""):
    # type: (pyNetLogo.NetLogoLink, str) -> float
    netlogo_link.command("setup")
    if post_setup_command:
        netlogo_link.command(post_setup_command)
        print("{0} executed".format(post_setup_command))

    metrics_dataframe = netlogo_link.repeat_report(
        netlogo_reporter=[TURTLE_PRESENT_REPORTER, EVACUATED_REPORTER, DEAD_REPORTER], reps=2000)  # type: pd.DataFrame

    evacuation_finished = metrics_dataframe[
        metrics_dataframe[TURTLE_PRESENT_REPORTER] == metrics_dataframe[DEAD_REPORTER]]

    evacuation_time = evacuation_finished.index.min()  # type: float
    print("evacuation time {0}".format(evacuation_time))

    return evacuation_time


def start_experiments(netlogo_home, netlogo_version, model_file, gui=True):
    netlogo_link = pyNetLogo.NetLogoLink(netlogo_home=netlogo_home,
                                         netlogo_version=netlogo_version,
                                         gui=gui)  # type: pyNetLogo.NetLogoLink
    netlogo_link.load_model(model_file)

    no_support_times = [run_simulation(netlogo_link)
                        for _ in range(SAMPLES)]  # type:List[float]
    support_times = [run_simulation(netlogo_link, post_setup_command=ENABLE_PASSENGER_COMMAND)
                     for _ in range(SAMPLES)]  # type:List[float]

    experiment_results = pd.DataFrame(data=list(zip(no_support_times, support_times)),
                                      columns=[NO_SUPPORT_COLUMN, ONLY_PASSENGER_SUPPORT_COLUMN])  # type:pd.DataFrame

    experiment_results.to_csv(RESULTS_CSV_FILE)
    print("Data written to {}".format(RESULTS_CSV_FILE))


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
    netlogo_model = "/home/cgc87/github/robot-assisted-evacuation/impact2.10.7/v2.11.0.nlogo"
    netlogo_directory = "/home/cgc87/netlogo-5.3.1-64"
    version = "5"

    start_experiments(netlogo_directory, version, netlogo_model, gui=False)

    plt.style.use('seaborn-darkgrid')
    analyse_results()
