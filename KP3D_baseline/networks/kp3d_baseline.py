import numpy as np
import torch
import torch.nn as nn
from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder
from keypoint_resnet import KeypointEncoder
from keypoint_resnet import KeypointDecoder
from KP3D_baseline.pose_estimation import PoseEstimation

class KP3D_Baseline(nn.Module):
    def __init__(self, options):
        super(KP3D_Baseline, self).__init__()
        self.opt = options
        self.depth_encoder = ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, self.opt.scales)
        self.keypoint_encoder = KeypointEncoder(self.opt.weights_init == "pretrained", self.opt.with_drop)
        self.keypoint_decoder = KeypointDecoder()
        self.pose_estimator = PoseEstimation(self.opt.K1, self.opt.K2)

    def forward(self):
        pass