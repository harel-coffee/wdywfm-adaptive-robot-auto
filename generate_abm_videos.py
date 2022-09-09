import glob
import natsort

from PIL import Image
from typing import List

from abm_analysis import NETLOGO_PROJECT_DIRECTORY

FRAME_FOLDER = NETLOGO_PROJECT_DIRECTORY + "impact2.10.7/frames"  # type:str


def main():
    frame_list = natsort.natsorted(glob.glob("{}/*png".format(FRAME_FOLDER)))  # type: List[str]
    print("Generating GIF for {} frames...".format(len(frame_list)))
    frames = [Image.open(frame_file) for frame_file in frame_list]  # type: List[Image]

    first_frame = frames[0]  # type: Image
    output_file = "img/evacuation_simulation.gif"  # type: str
    frame_duration = 200  # type:int
    first_frame.save(output_file, format="GIF", append_images=frames,
                     save_all=True, duration=frame_duration)
    print("Animation generated at {}".format(output_file))


if __name__ == "__main__":
    main()
