import datetime
import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, OptimizeResult
from scipy.special import logit, expit
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from typing import Optional, Callable

from analyser import SyntheticTypeAnalyser


def plot_reliability_diagram(person_type, person_type_probabilities, bins, probability_label):
    bin_true_probability, bin_predicted_probability = calibration_curve(person_type,
                                                                        person_type_probabilities,
                                                                        n_bins=bins)

    calibration_error = calculate_ece_from_calibration_curve(bin_true_probability, bin_predicted_probability,
                                                             person_type_probabilities)  # type: float
    brier_score = brier_score_loss(bin_true_probability, bin_predicted_probability)  # type: float
    logging.info("Expected Calibration Error: {}. Brier score: {}".format(calibration_error, brier_score))

    plt.hist(person_type_probabilities,
             weights=np.ones_like(person_type_probabilities) / len(person_type_probabilities),
             alpha=.4, bins=np.maximum(10, bins))
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
        start_isotonic_regression(type_analyser, sensor_data, person_type, bins)

        # start_plat_scalling(person_type, person_type_probabilities, bins)


def start_isotonic_regression(type_analyser, sensor_data_validation, person_type_validation, bins, method="isotonic"):
    # type: (SyntheticTypeAnalyser, np.ndarray, np.ndarray, int, str) -> None

    person_type_probabilities = type_analyser.obtain_probabilities(sensor_data_validation)  # type: np.ndarray
    plot_reliability_diagram(person_type_validation, person_type_probabilities, bins,
                             probability_label="after_training")

    keras_classifier = type_analyser.keras_classifier
    calibrated_classifier = CalibratedClassifierCV(base_estimator=keras_classifier, cv="prefit",
                                                   method=method)  # type: CalibratedClassifierCV
    calibrated_classifier.fit(sensor_data_validation, person_type_validation)

    # We need to try this later over the test dataset
    calibrated_probabilities = calibrated_classifier.predict_proba(sensor_data_validation)[:, 1]  # type: np.ndarray
    plot_reliability_diagram(person_type_validation, calibrated_probabilities, bins,
                             probability_label="after_calibration")


def start_plat_scalling(person_type, person_type_probabilities, bins):
    # type: (np.ndarray, np.ndarray, int) -> None

    calibration_object = fit_platt_scaling_calibration(person_type, person_type_probabilities)  # type: Callable
    calibrated_probabilities = calibration_object(person_type_probabilities)

    new_true_probability, new_predicted_probability = calibration_curve(person_type,
                                                                        calibrated_probabilities,
                                                                        n_bins=bins)
    logging.info("NEW: Expected Calibration Error: {}".format(
        calculate_ece_from_calibration_curve(new_true_probability, new_predicted_probability,
                                             person_type_probabilities)))
    plt.plot(new_predicted_probability, new_true_probability, label="Plat scalling")


def fit_platt_scaling_calibration(person_type, person_type_probabilities):
    # type: (np.ndarray, np.ndarray) -> Callable
    """
    Code taken from: http://kdd2020.nplan.io/
    """
    person_type_logits = logit(person_type_probabilities)  # type: np.ndarray

    def scalar_function(x, *args):
        a, b = x
        y_logit_scaled = a * person_type_logits + b
        y_pred_inner = expit(y_logit_scaled)
        bce = sum(
            [-(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p)) for y_t, y_p in
             zip(person_type, y_pred_inner)
             if not y_p == 0])

        return bce

    optimisation_result = minimize(fun=scalar_function,
                                   x0=[1, 0],
                                   method="Nelder-Mead",
                                   options={'xatol': 1e-8,
                                            'disp': True})  # type: OptimizeResult

    def calibrate(person_type_prob_input):
        # type: (np.ndarray) -> np.ndarray
        logits_from_input = logit(person_type_prob_input)  # type: np.ndarray
        return expit(optimisation_result.x[0] * logits_from_input + optimisation_result.x[1])

    return calibrate


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
