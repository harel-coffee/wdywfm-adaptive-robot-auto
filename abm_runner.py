import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from typing import Dict

from analyser import SyntheticTypeAnalyser
from controller import AutonomicManagerController
from environment import NetlogoEvacuationEnvironment
from synthetic_runner import MODEL_FILE, ENCODER_FILE

PROJECT_DIRECTORY = "/home/cgc87/github/wdywfm-adaptive-robot/"  # type:str


def run_scenario(robot_controller, emergency_environment):
    # type: ( AutonomicManagerController,  NetlogoEvacuationEnvironment) -> str

    return "ask-staff"


def main():
    parser = ArgumentParser("Get a robot action from the adaptive controller",
                            formatter_class=ArgumentDefaultsHelpFormatter)  # type: ArgumentParser
    parser.add_argument("helper_gender")
    parser.add_argument("helper_culture")
    parser.add_argument("helper_age")
    parser.add_argument("fallen_gender")
    parser.add_argument("fallen_culture")
    parser.add_argument("fallen_age")

    arguments = parser.parse_args()
    configuration = vars(arguments)  # type:Dict

    type_analyser = SyntheticTypeAnalyser(model_file=PROJECT_DIRECTORY + MODEL_FILE)  # type: SyntheticTypeAnalyser
    robot_controller = AutonomicManagerController(type_analyser)

    emergency_environment = NetlogoEvacuationEnvironment(configuration,
                                                         PROJECT_DIRECTORY + ENCODER_FILE)  # type: NetlogoEvacuationEnvironment

    robot_action = run_scenario(robot_controller, emergency_environment)  # type:str
    print(robot_action)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
