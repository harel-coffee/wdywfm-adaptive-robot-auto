import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from synthetic_runner import train_type_analyser

NETLOGO_DATA_FILE_PREFIX = "request-for-help-results"  # type:str
REQUEST_RESULT_COLUMN = "offer-help"  # type:str


def get_netlogo_dataset():
    # type:() -> Tuple[np.ndarray, np.ndarray]

    dataframes = [pd.read_csv("data/{}_{}.csv".format(dataframe_index, NETLOGO_DATA_FILE_PREFIX))
                  for dataframe_index in range(0, 97)]  # type: List[pd.DataFrame]

    netlogo_dataframe = pd.concat(dataframes, axis=0)  # type: pd.DataFrame

    netlogo_sensor_data = netlogo_dataframe.drop(REQUEST_RESULT_COLUMN, axis=1)  # type: pd.DataFrame
    netlogo_person_type = netlogo_dataframe[REQUEST_RESULT_COLUMN]  # type: pd.DataFrame

    return netlogo_sensor_data.values, netlogo_person_type.values


def main():
    target_accuracy = None
    max_epochs = 500  # type: int
    encode_categorical_data = True  # type: bool
    training_batch_size = 100

    sensor_data, person_type = get_netlogo_dataset()
    sensor_data_train, sensor_data_test, person_type_train, person_type_test = train_test_split(sensor_data,
                                                                                                person_type,
                                                                                                test_size=0.33,
                                                                                                random_state=0)
    _ = train_type_analyser(sensor_data_train, person_type_train, training_batch_size, target_accuracy,
                            encode_categorical_data, max_epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
