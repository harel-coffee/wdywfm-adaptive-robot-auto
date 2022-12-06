import datetime
import logging

import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss
from typing import Optional

from analyser import SyntheticTypeAnalyser


def plot_reliability_diagram(sensor_data, person_type, model_file):
    # type: (np.ndarray, np.ndarray, str) -> None

    logging.info("Reliability diagram for model {}.".format(model_file))

    type_analyser = SyntheticTypeAnalyser(model_file=model_file)  # type: SyntheticTypeAnalyser

    person_type_predictions = type_analyser.predict_type(sensor_data)  # type: np.ndarray
    person_type_probabilities = type_analyser.obtain_probabilities(sensor_data)

    accuracy = accuracy_score(person_type, person_type_predictions)
    loss = log_loss(person_type, person_type_probabilities)
    logging.info("Accuracy {}, log loss {}".format(accuracy, loss))

    bin_true_probability, bin_predicted_probability = calibration_curve(person_type,
                                                                        person_type_probabilities)
    logging.info("Expected Calibration Error: {}".format(
        calculate_ece_from_calibration_curve(bin_true_probability, bin_predicted_probability,
                                             person_type_probabilities)))

    plt.plot([0, 1], [0, 1], color="#FE4A49", linestyle=":", label="Perfectly calibrated model")
    plt.plot(bin_predicted_probability, bin_true_probability, "s-", label=model_file, color="#162B37")

    suffix = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")  # type: str
    plt.title("Reliability diagram: {}".format(suffix))
    plt.ylabel("Fraction of positives", )
    plt.xlabel("Mean predicted value")
    plt.grid(True, color="#B2C7D9")

    plt.savefig("img/calibration_curve_{}.png".format(suffix),
                bbox_inches='tight', pad_inches=0)
    plt.show()


def get_expected_calibration_error(sensor_data, person_type, model_file=None, type_analyser=None):
    # type: (np.ndarray, np.ndarray, Optional[str], Optional[SyntheticTypeAnalyser]) -> float

    if model_file is not None:
        logging.info("Expected calibration error for model {}.".format(model_file))
        type_analyser = SyntheticTypeAnalyser(model_file=model_file)  # type: SyntheticTypeAnalyser
    else:
        logging.info("Expected calibration error for SyntheticTypeAnalyser instance.")

    person_type_probabilities = type_analyser.obtain_probabilities(sensor_data)  # type: np.ndarray

    bin_true_probability, bin_predicted_probability = calibration_curve(person_type,
                                                                        person_type_probabilities)

    result = calculate_ece_from_calibration_curve(bin_true_probability, bin_predicted_probability,
                                                  person_type_probabilities)  # type: float
    logging.info("Result: {}".format(result))
    return result


def calculate_ece_from_calibration_curve(bin_true_probability, bin_predicted_probability, person_type_probabilities):
    # type:(np.ndarray, np.ndarray, np.ndarray) -> float
    number_of_bins = len(bin_true_probability)  # type: int
    histogram = np.histogram(a=person_type_probabilities, range=(0, 1), bins=number_of_bins)
    bin_sizes = histogram[0]
    result = 0.0  # type: float

    total_samples = float(sum(bin_sizes))  # type: float
    for bin_index in np.arange(len(bin_sizes)):
        current_bin_size = bin_sizes[bin_index]  # type: int
        true_probability = bin_true_probability[bin_index]  # type: float
        predicted_probability = bin_predicted_probability[bin_index]  # type: float

        result += current_bin_size / total_samples * np.abs(true_probability - predicted_probability)

    return result
