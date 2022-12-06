import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from abm_analysis import run_parallel_simulations, SET_STAFF_SUPPORT_COMMAND, SET_PASSENGER_SUPPORT_COMMAND, \
    SET_FALL_LENGTH_COMMAND
from prob_calibration import plot_reliability_diagram
from synthetic_runner import TYPE_ANALYSER_MODEL_FILE, encode_training_data, plot_confusion_matrix, plot_training, \
    train_type_analyser

MAX_EPOCHS = 500  # type: int
EARLY_STOPPING_PATIENCE = int(MAX_EPOCHS * 0.15)  # type: int
TRAINING_BATCH_SIZE = 128  # type: int
LEARNING_RATE = 0.0001  # type: float
UNITS_PER_LAYER = [64, 64, 64]  # type: List[int]

TRAINING_DATA_DIRECTORY = "data/training"
NETLOGO_DATA_FILE_PREFIX = "request-for-help-results"  # type:str
REQUEST_RESULT_COLUMN = "offer-help"  # type:str

SIMULATION_RUNS = 100  # type:int
FALL_LENGTH = 30  # type:int

ENABLE_DATA_COLLECTION_COMMAND = "set ENABLE_DATA_COLLECTION TRUE"  # type:str
DISABLE_LOGGING_COMMAND = "set ENABLE_LOGGING FALSE"  # type:str
DISABLE_FRAME_GENERATION_COMMAND = "  set ENABLE_FRAME_GENERATION FALSE"  # type:str

CONFIGURATION_COMMANDS = [
    SET_STAFF_SUPPORT_COMMAND.format("FALSE"),
    SET_PASSENGER_SUPPORT_COMMAND.format("FALSE"),
    SET_FALL_LENGTH_COMMAND.format(FALL_LENGTH),
    ENABLE_DATA_COLLECTION_COMMAND,
    DISABLE_LOGGING_COMMAND,
    DISABLE_FRAME_GENERATION_COMMAND,
]  # type: List[str]


def get_netlogo_dataset():
    # type:() -> Tuple[np.ndarray, np.ndarray]

    dataframes = [pd.read_csv("{}/{}_{}.csv".format(TRAINING_DATA_DIRECTORY,
                                                    dataframe_index,
                                                    NETLOGO_DATA_FILE_PREFIX))
                  for dataframe_index in range(0, 97)]  # type: List[pd.DataFrame]

    netlogo_dataframe = pd.concat(dataframes, axis=0)  # type: pd.DataFrame

    netlogo_sensor_data = netlogo_dataframe.drop(REQUEST_RESULT_COLUMN, axis=1)  # type: pd.DataFrame
    netlogo_person_type = netlogo_dataframe[REQUEST_RESULT_COLUMN]  # type: pd.DataFrame

    return netlogo_sensor_data.values, netlogo_person_type.values


def generate_training_data(simulation_runs=None, configuration_commands=None):
    # type: (int, List[str]) -> None
    print("Generating training data from {} simulation runs".format(SIMULATION_RUNS))
    _ = run_parallel_simulations(simulation_runs, configuration_commands, gui=False)


def start_training(max_epochs, training_batch_size, learning_rate, units_per_layer,
                   early_stopping_patience):
    # type: (int, int,float, List[int], int) -> None

    target_accuracy = None
    under_sample = False  # type: bool
    calculate_weights = True  # type: bool

    sensor_data, person_type = get_netlogo_dataset()  # type: Tuple[np.ndarray, np.ndarray]
    sensor_data_training, sensor_data_test, person_type_training, person_type_test = train_test_split(sensor_data,
                                                                                                      person_type,
                                                                                                      test_size=0.33,
                                                                                                      random_state=0)
    sensor_data_training = encode_training_data(sensor_data_training)  # type:np.ndarray

    sensor_data_training, sensor_data_validation, person_type_training, person_type_validation = train_test_split(
        sensor_data_training,
        person_type_training,
        stratify=person_type_training,
        test_size=0.33)

    training_history = train_type_analyser(sensor_data_training, person_type_training,
                                           sensor_data_validation, person_type_validation,
                                           training_batch_size, target_accuracy,
                                           units_per_layer, max_epochs,
                                           learning_rate=learning_rate,
                                           patience=early_stopping_patience,
                                           balance_data=under_sample,
                                           calculate_weights=calculate_weights)
    plot_training(training_history, metric="binary_crossentropy")
    plot_training(training_history, metric="acc")

    plot_reliability_diagram(sensor_data_validation, person_type_validation, TYPE_ANALYSER_MODEL_FILE)
    plot_confusion_matrix(sensor_data_validation, person_type_validation, TYPE_ANALYSER_MODEL_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # generate_training_data(SIMULATION_RUNS, CONFIGURATION_COMMANDS)
    start_training(MAX_EPOCHS, TRAINING_BATCH_SIZE, LEARNING_RATE, UNITS_PER_LAYER,
                   EARLY_STOPPING_PATIENCE)
