import os
import time
import json
import yaml
import torch
import math
import cv2
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils import ui_utils
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import draw_box, clear_debug_drawing, draw_line

from scipy.spatial.transform import Rotation as R
from get_video import create_video_from_images

def main():
    pass

if __name__ == "__main__":
    main()