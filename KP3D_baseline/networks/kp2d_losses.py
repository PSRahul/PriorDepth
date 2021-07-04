import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

from utils.image import (image_grid, to_color_normalized,
                              to_gray_normalized)

def calculate_2d_warping_loss(inputs,outputs):
    return 0