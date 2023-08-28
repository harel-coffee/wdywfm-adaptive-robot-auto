import datetime
import logging
import time

import numpy as np
from matplotlib import pyplot as plt
from numpy import load
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from typing import Optional, Union, Tuple

from analyser import SyntheticTypeAnalyser, CalibratedTypeAnalyser


def plot_reliability_diagram(person_type, person_type_probabilities, bins, probability_label):
    bin_true_probability, bin_predicted_probability = calibration_curve(person_type,
                                                                        person_type_probabilities,
                                                                        n_bins=bins)

    calibration_error = calculate_ece_from_calibration_curve(bin_true_probability, bin_predicted_probability,
                                                             person_type_probabilities)  # type: float
    brier_score = brier_score_loss(bin_true_probability, bin_predicted_probability)  # type: float
    roc_auc = roc_auc_score(person_type, person_type_probabilities)  # type: float
    log_loss_value = log_loss(person_type, person_type_probabilities)  # type:float
    logging.info(
        "Expected Calibration Error: {}. Brier score: {}. ROC AUC: {}. Log loss: {}".format(calibration_error,
                                                                                            brier_score, roc_auc,
                                                                                            log_loss_value))

    plt.hist(person_type_probabilities,
             weights=np.ones_like(person_type_probabilities) / len(person_type_probabilities),
             alpha=.4, bins=bins)
    plt.plot([0, 1], [0, 1], color="#FE4A49", linestyle=":", label="Perfectly calibrated model")
    plt.plot(bin_predicted_probability, bin_true_probability, "s-", label=probability_label, color="#162B37")

    suffix = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")  # type: str
    plt.title("RD: {} (ECE {:.2f}, BS {:.2f})".format(suffix, calibration_error, brier_score))
    plt.ylabel("Fraction of positives", )
    plt.xlabel("Mean predicted value")
    plt.legend()
    plt.grid(True, color="#B2C7D9")
    plt.savefig("img/calibration_curve_{}.png".format(suffix),
                bbox_inches='tight', pad_inches=0)
    plt.show()


def start_calibration(sensor_data, person_type, model_file, calibrate=False, bins=5):
    # type: (np.ndarray, np.ndarray, str, bool, int) -> None

    logging.info("Reliability diagram for model {}.".format(model_file))

    type_analyser = SyntheticTypeAnalyser(model_file=model_file)  # type: SyntheticTypeAnalyser

    person_type_probabilities = type_analyser.obtain_probabilities(sensor_data)  # type: np.ndarray

    plot_reliability_diagram(person_type, person_type_probabilities, bins, probability_label=model_file)

    if calibrate:
        # start_isotonic_regression(bin_true_probability, bin_predicted_probability)
        start_probability_calibration(type_analyser, sensor_data, person_type, bins)

        # start_plat_scalling(person_type, person_type_probabilities, bins)


def get_calibrated_model(type_analyser, calibration_sensor_data_file, calibration_person_type_file, method):
    # type: (SyntheticTypeAnalyser, str, str, str) ->  Tuple[CalibratedTypeAnalyser, np.ndarray, np.ndarray]

    sensor_data_validation = load(calibration_sensor_data_file)  # type:np.ndarray
    logging.info("Calibration sensor data loaded from: {}".format(calibration_sensor_data_file))
    logging.info("Validation samples: {}".format(sensor_data_validation.shape[0]))

    person_type_validation = load(calibration_person_type_file)  # type:np.ndarray
    logging.info("Calibration person type data loaded from: {}".format(calibration_sensor_data_file))

    keras_classifier = type_analyser.keras_classifier
    calibrated_classifier = CalibratedTypeAnalyser(keras_classifier, method)  # type: CalibratedTypeAnalyser
    calibrated_classifier.train(sensor_data_validation, person_type_validation)

    return calibrated_classifier, sensor_data_validation, person_type_validation


def start_probability_calibration(type_analyser, calibration_sensor_data_file, calibration_person_type_file,
                                  sensor_data_test, person_type_test, bins, method="isotonic"):
    # type: (Union[SyntheticTypeAnalyser, str], str, str, np.ndarray, np.ndarray, int, str) -> None

    calibration_start_time = time.time()  # type: float

    if isinstance(type_analyser, basestring):
        type_analyser = SyntheticTypeAnalyser(model_file=type_analyser)  # type:SyntheticTypeAnalyser

    logging.info(str(type_analyser))
    calibrated_classifier, sensor_data_validation, person_type_validation = get_calibrated_model(
        type_analyser,
        calibration_sensor_data_file,
        calibration_person_type_file,
        method)
    logging.info("Finished calibrating probabilities after {} seconds".format(time.time() - calibration_start_time))

    person_type_probabilities = type_analyser.obtain_probabilities(sensor_data_validation)  # type: np.ndarray
    plot_reliability_diagram(person_type_validation, person_type_probabilities, bins,
                             probability_label="after_training")

    logging.info("Calibrating probabilities using: {} method".format(method))
    calibrated_probabilities_validation = calibrated_classifier.obtain_probabilities(
        sensor_data_validation)  # type: np.ndarray
    plot_reliability_diagram(person_type_validation, calibrated_probabilities_validation, bins,
                             probability_label="after_calibration_validation")

    # We need to try this later over the test dataset
    logging.info("Testing samples: {}".format(sensor_data_test.shape[0]))
    calibrated_probabilities_test = calibrated_classifier.obtain_probabilities(sensor_data_test)  # type: np.ndarray
    plot_reliability_diagram(person_type_test, calibrated_probabilities_test, bins,
                             probability_label="after_calibration_test")


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
