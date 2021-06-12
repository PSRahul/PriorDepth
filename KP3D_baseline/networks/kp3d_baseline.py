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
        self.pose_estimator = PoseEstimation(K1, K2, self.opt.no_cuda)
        ## TODO: // add K1 and K2 to options! or check whether K is correct in trainer.py line 36

    def reshape_kp2d_preds(self, kp_output, i):
        score, coord, feat = kp_output[0], kp_output[1], kp_output[2]
        # TODO: we need to filter based on scores and then reshape these vectors for pose estimation
        # TODO: for instance match_mnn in pose_estimation requires len(desc1.shape) == 2
        # TODO: need to figure out how to filter and reshape these data
        # Score map (B, 1, H_out, W_out)
        # Keypoint coordinates (B, 2, H_out, W_out)
        # Keypoint descriptors (B, 256, H_out, W_out)

        mask = score[:, 0, :, :] > 0.7
        coord_mask = torch.stack((mask, mask), dim=1)
        desc_mask = torch.stack(256*[mask], dim=1)

        coord = coord[coord_mask]
        coord = coord.reshape((2, coord.shape[0] // 2))

        feat = feat[desc_mask]
        feat = feat.reshape((256, feat.shape[0] // 256))

        return {'kp{}_score'.format(i): score, 'kp{}_coord'.format(i): coord, 'kp{}_feat'.format(i):  feat}

    def forward(self, input_image):
        # TODO: make sure color, 0, 0 is target image and color, 1, 0 is context image
        # TODO: calculate transformation matrix from target to context
        # TODO: warp target pixels to obtain context pixels
        # TODO: compute photometric loss between target image and warped target image
        outputs = {}
        print('in forward kp3d')
        depth_features = self.depth_encoder(input_image["color_aug", 0, 0])
        disp_outputs = self.depth_decoder(depth_features)
        outputs.update(disp_outputs)

        kp2d_output1 = self.keypoint_net(input_image["color_aug", 0, 0])
        kp2d_output2 = self.keypoint_net(input_image["color_aug", 1, 0])

        kp2d_output1 = self.reshape_kp2d_preds(kp2d_output1, 1)
        kp2d_output2 = self.reshape_kp2d_preds(kp2d_output2, 2)

        if kp2d_output1['kp1_coord'].shape[1] > kp2d_output2['kp2_coord'].shape[1]:
            missing_num_dim = kp2d_output1['kp1_coord'].shape[1] - kp2d_output2['kp2_coord'].shape[1]
            zeros_coord_array = torch.zeros((2, missing_num_dim), device=torch.device("cpu" if self.opt.no_cuda else "cuda"))
            zeros_desc_array = torch.zeros((256, missing_num_dim), device=torch.device("cpu" if self.opt.no_cuda else "cuda"))
            kp2d_output2['kp2_coord'] = torch.cat((kp2d_output2['kp2_coord'], zeros_coord_array), dim=1)
            kp2d_output2['kp2_feat'] = torch.cat((kp2d_output2['kp2_feat'], zeros_desc_array), dim=1)
        else:
            missing_num_dim = kp2d_output2['kp2_coord'].shape[1] - kp2d_output1['kp1_coord'].shape[1]
            zeros_coord_array = torch.zeros((2, missing_num_dim), device=torch.device("cpu" if self.opt.no_cuda else "cuda"))
            zeros_desc_array = torch.zeros((256, missing_num_dim), device=torch.device("cpu" if self.opt.no_cuda else "cuda"))
            kp2d_output1['kp1_coord'] = torch.cat((kp2d_output1['kp1_coord'], zeros_coord_array), dim=1)
            kp2d_output1['kp1_feat'] = torch.cat((kp2d_output1['kp1_feat'], zeros_desc_array), dim=1)

        outputs.update(kp2d_output1)
        outputs.update(kp2d_output2)

        R, t = self.pose_estimator.get_pose(kp2d_output1['kp1_coord'].T, kp2d_output2['kp2_coord'].T,
                                            kp2d_output1['kp1_feat'].T, kp2d_output2['kp2_feat'].T)

        outputs["R"] = R
        outputs["t"] = t
        return outputs
