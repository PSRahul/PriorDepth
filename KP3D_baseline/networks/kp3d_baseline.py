import torch.nn as nn
from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_estimation import PoseEstimation
from .keypoint_net import KeypointNet
import torch


class KP3D_Baseline(nn.Module):
    def __init__(self, options, K1, K2):
        super(KP3D_Baseline, self).__init__()
        self.opt = options
        self.depth_encoder = ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, self.opt.scales)
        # self.keypoint_encoder = KeypointEncoder(self.opt.weights_init == "pretrained", self.opt.with_drop)
        # self.keypoint_decoder = KeypointDecoder()
        self.keypoint_net = KeypointNet()
        self.pose_estimator = PoseEstimation(K1, K2)
        ## TODO: // add K1 and K2 to options! or check whether K is correct in trainer.py line 36

    def reshape_kp2d_preds(self, kp_output):
        score, coord, feat = kp_output[0], kp_output[1], kp_output[2]
        # TODO: we need to filter based on scores and then reshape these vectors for pose estimation
        # TODO: for instance match_mnn in pose_estimation requires len(desc1.shape) == 2
        # TODO: need to figure out how to filter and reshape these data
        # Score map (B, 1, H_out, W_out)
        # Keypoint coordinates (B, 2, H_out, W_out)
        # Keypoint descriptors (B, 256, H_out, W_out)
        print(score.shape)
        print(score > 0.5)
        print(torch.nonzero(score > 0.5).shape)

        return score, coord, feat

    def forward(self, input_image):
        outputs = {}
        # print(input_image)
        print('in model forward')
        print(input_image["color_aug", 0, 0].shape)
        print(input_image["color_aug", 1, 0].shape)
        depth_features = self.depth_encoder(input_image["color_aug", 0, 0])
        print('after depth encoder')
        print(depth_features[0].shape)
        disp_outputs = self.depth_decoder(depth_features)
        print('after depth decoder')
        print(disp_outputs[("disp", 0)].shape)
        outputs.update(disp_outputs)

        kp2d_output1 = self.keypoint_net(input_image["color_aug", 0, 0])
        kp2d_output2 = self.keypoint_net(input_image["color_aug", 1, 0])

        print('after keypoint decoder')
        kp2d_output1 = self.reshape_kp2d_preds(kp2d_output1)
        kp2d_output2 = self.reshape_kp2d_preds(kp2d_output2)

        outputs.update(kp2d_output1)
        outputs.update(kp2d_output2)

        R, t = self.pose_estimator.get_pose(kp2d_output1[1], kp2d_output2[1], kp2d_output1[2], kp2d_output2[2])
        print('after pose estimate')
        print(R)
        print(t)
        outputs["R"] = R
        outputs["t"] = t
        return outputs
