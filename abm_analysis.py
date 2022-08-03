import pyNetLogo


def run_simulation(netlogo_link):
    netlogo_link.command("setup")
    evacuation_time = netlogo_link.repeat_report(netlogo_reporter="current_time",
                                                 reps=2000)
    print(evacuation_time)

    return evacuation_time


def main(netlogo_home, netlogo_version, model_file):
    netlogo_link = pyNetLogo.NetLogoLink(netlogo_home=netlogo_home,
                                         netlogo_version=netlogo_version,
                                         gui=True)
    netlogo_link.load_model(model_file)

    run_simulation(netlogo_link)


if __name__ == "__main__":
    netlogo_model = "/home/cgc87/github/robot-assisted-evacuation/impact2.10.7/v2.11.0.nlogo"
    netlogo_directory = "/home/cgc87/netlogo-5.3.1-64"
    version = "5"

    main(netlogo_directory, version, netlogo_model)
