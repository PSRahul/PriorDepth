import torch.nn as nn
from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .keypoint_resnet import KeypointEncoder
from .keypoint_resnet import KeypointDecoder
from .pose_estimation import PoseEstimation


class KP3D_Baseline(nn.Module):
    def __init__(self, options, K1, K2):
        super(KP3D_Baseline, self).__init__()
        self.opt = options
        self.depth_encoder = ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, self.opt.scales)
        self.keypoint_encoder = KeypointEncoder(self.opt.weights_init == "pretrained", self.opt.with_drop)
        self.keypoint_decoder = KeypointDecoder()
        self.pose_estimator = PoseEstimation(K1, K2)
        ## TODO: // add K1 and K2 to options!

    def forward(self, input_image):
        outputs = {}

        depth_features = self.depth_encoder(input_image)
        disp_outputs = self.depth_decoder(depth_features)
        outputs.update(disp_outputs)

        kp_input_features = self.keypoint_encoder(input_image)
        kp_features = self.keypoint_decoder(kp_input_features)
        outputs.update(kp_features)

        R, t = self.pose_estimator.get_pose(kp_features["feature"])
        outputs["R"] = R
        outputs["t"] = t
        return outputs
