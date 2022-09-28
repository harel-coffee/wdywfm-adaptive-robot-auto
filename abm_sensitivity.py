import itertools

import pandas as pd
import seaborn
from typing import List, Dict

from abm_analysis import SIMULATION_SCENARIOS, ADAPTIVE_SUPPORT_COLUMN, run_parallel_simulations, PLOT_STYLE, \
    SET_FALL_LENGTH_COMMAND
import matplotlib.pyplot as plt

EVACUATION_TIME_COLUMN = "evacuation_time"

PASSENGER_BONUS_COLUMN = "passenger_bonus"  # type:str
FALL_LENGTH_COLUMN = "fall_length"  # type:str

FALL_LENGTHS = range(60, 660, 60)  # type: List[int]
PASSENGER_BONUS = range(5, 55, 5)  # type: List[int]
SENSITIVITY_SAMPLES = 12  # type: int
SENSITIVITY_DATA_FILE = "data/sensitivity_analysis.csv"  # type:str

SET_PASSENGER_BONUS_COMMAND = "set ROBOT_REQUEST_BONUS {}"  # type:str


def get_heatmap(annotated=False):
    experiment_data = pd.read_csv(SENSITIVITY_DATA_FILE, index_col=[0])  # type: pd.DataFrame
    experiment_data = experiment_data.dropna()

    consolidated_data = experiment_data.groupby(
        [PASSENGER_BONUS_COLUMN, FALL_LENGTH_COLUMN]).median()  # type: pd.DataFrame

    heatmap_data = consolidated_data.pivot_table(values=EVACUATION_TIME_COLUMN,
                                                 index=FALL_LENGTH_COLUMN,
                                                 columns=PASSENGER_BONUS_COLUMN)

    print(heatmap_data)

    _ = seaborn.heatmap(heatmap_data, annot=annotated, fmt=".1f")
    plt.savefig("img/sensitivity_analysis.eps", format="eps")
    plt.savefig("img/sensitivity_analysis.png", format="png")

    plt.show()


def generate_data_for_analysis():
    sensitivity_data = []  # type: List[Dict[str, float]]

    for fall_length, passenger_bonus in itertools.product(FALL_LENGTHS, PASSENGER_BONUS):

        passenger_bonus = passenger_bonus / 100.0  # type:float
        print("fall_length {} passenger_bonus {}".format(fall_length, passenger_bonus))

        post_setup_commands = SIMULATION_SCENARIOS[ADAPTIVE_SUPPORT_COLUMN]  # type: List[str]
        post_setup_commands.append(SET_FALL_LENGTH_COMMAND.format(fall_length))
        post_setup_commands.append(SET_PASSENGER_BONUS_COMMAND.format(passenger_bonus))

        scenario_times = run_parallel_simulations(SENSITIVITY_SAMPLES,
                                                  post_setup_commands=post_setup_commands)  # type: List[float]
        for evacuation_time in scenario_times:
            sensitivity_data.append({FALL_LENGTH_COLUMN: fall_length,
                                     PASSENGER_BONUS_COLUMN: passenger_bonus,
                                     EVACUATION_TIME_COLUMN: evacuation_time})

    results_dataframe = pd.DataFrame(sensitivity_data)
    results_dataframe.to_csv(SENSITIVITY_DATA_FILE)

    print("Results written to {}".format(SENSITIVITY_DATA_FILE))


if __name__ == "__main__":
    plt.style.use(PLOT_STYLE)
    # generate_data_for_analysis()
    get_heatmap(annotated=False)
