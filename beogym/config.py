"""
This file contains configuration variables
"""
import os

# CAMERA_IMAGES_PATH is the variable that points to the images for the
# camera images.
# Server
# CAMERA_IMAGES_PATH = "/home/realnav/test_dataset/lb3"

# Locally
CAMERA_IMAGES_PATH = "/lab/kiran/data/pano_images"
GPS_DATA_PATH = "/lab/kiran/data/pano_gps.csv"
PANO_IMAGE_LIMIT = 0
PANO_IMAGE_RESOLUTION = (208, 1664,)
PANO_HOV = 208
############################################################


# Utility methods to navigate the map manually
# The Actioned defined below map keyboard inputs to the actions that can be taken on
# the panoramas.
class KeyBoardActions:
    LEFT = ord('a')
    RIGHT = ord('d')
    FORWARD = ord('w')
    REVERSE = ord('s')
    KILL_PROGRAM = ord('k')


# Directions can be entered into the terminal in headless mode
class InputActions:
    LEFT = "left"
    RIGHT = "right"
    FORWARD = "forward"
    REVERSE = "reverse"
    KILL_PROGRAM = "quit"


# List of all actions related to moving in a specific direction
DIRECTION_ACTIONS = [KeyBoardActions.LEFT, KeyBoardActions.RIGHT, KeyBoardActions.FORWARD, KeyBoardActions.REVERSE,
                     InputActions.LEFT, InputActions.RIGHT, InputActions.FORWARD, InputActions.REVERSE]


class ConfigModes:
    PANO = "pano"
    HEADLESS = "headless"
    RANDOM = "random"
    COMP = "comp"
    HUMAN = "human"
    SPPLANNER = "spplanner"


# image mode should either be pano or something else
PANO_IMAGE_MODE = True

# Whether or not to show images live
HEADLESS_MODE = False

# How the beogym API should be interacted with.
INTERACTION_MODE = ConfigModes.RANDOM
#INTERACTION_MODE = ConfigModes.HUMAN

# Stores the traversed path
SAVE_IMAGE_PATH = False

# ../_tmp should be in the directory at the same level as src/.
# This should only be a relative path so it does not delete anything sensitive by accident.
# Only is used if SAVE_IMAGE_PATH is set to True
IMAGE_PATH_DIR = "../_tmp/traversed_path"

IMAGE_SOURCE_EXTENSION = "jpg"

# Verify that the image path is not an absolute path
if os.path.isabs(IMAGE_PATH_DIR):
    raise ValueError("IMAGE_PATH_DIR should be relative")




