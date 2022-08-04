import pyNetLogo
import pandas as pd

TURTLE_PRESENT_REPORTER = "count turtles"
EVACUATED_REPORTER = "number_passengers - count agents + 1"
DEAD_REPORTER = "count agents with [ st_dead = 1 ]"


def run_simulation(netlogo_link):
    # type: (pyNetLogo.NetLogoLink) -> float
    netlogo_link.command("setup")
    metrics_dataframe = netlogo_link.repeat_report(
        netlogo_reporter=[TURTLE_PRESENT_REPORTER, EVACUATED_REPORTER, DEAD_REPORTER], reps=2000)  # type: pd.DataFrame

    print(metrics_dataframe)

    evacuation_finished = metrics_dataframe[
        metrics_dataframe[TURTLE_PRESENT_REPORTER] == metrics_dataframe[DEAD_REPORTER]]

    evacuation_time = evacuation_finished.index.min()  # type: float

    return evacuation_time


def main(netlogo_home, netlogo_version, model_file):
    netlogo_link = pyNetLogo.NetLogoLink(netlogo_home=netlogo_home,
                                         netlogo_version=netlogo_version,
                                         gui=True)
    netlogo_link.load_model(model_file)

    evacuation_time = run_simulation(netlogo_link)  # type:float
    print("evacuation_time {0}".format(evacuation_time))


if __name__ == "__main__":
    netlogo_model = "/home/cgc87/github/robot-assisted-evacuation/impact2.10.7/v2.11.0.nlogo"
    netlogo_directory = "/home/cgc87/netlogo-5.3.1-64"
    version = "5"

    main(netlogo_directory, version, netlogo_model)
