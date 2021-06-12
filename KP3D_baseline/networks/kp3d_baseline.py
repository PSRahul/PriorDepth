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

    def batch_reshape_kp2d_preds(self, kp2d_output,threshold=0.3):
        score, coord, feat = kp2d_output
        org_shape=score.shape
        new_shape=(org_shape[0],org_shape[2]*org_shape[3],org_shape[1])
        score, coord, feat = score.permute(0,2,3,1),coord.permute(0,2,3,1),feat.permute(0,2,3,1)
        score, coord, feat = score.reshape((new_shape[0],new_shape[1])),coord.reshape((new_shape[0],new_shape[1],2)),feat.reshape((new_shape[0],new_shape[1],256))
        

        score_filtered=torch.zeros((new_shape[0],new_shape[1]))
        coord_filtered=torch.zeros((new_shape[0],new_shape[1],2))
        feat_filtered=torch.zeros((new_shape[0],new_shape[1],256))

        #print("Score Shape",score.shape)
        #print("Coordinate Shape",coord.shape)
        #print("Descriptor Shape",feat.shape)

        for i in range(new_shape[0]):
            score_i=score[i,:]
            score_i_index=torch.argsort(score_i,descending=True)
            score_filtered[i,:]=score_i[score_i_index]
            coord_filtered[i,:,:]=coord[i,score_i_index,:]
            feat_filtered[i,:,:]=feat[i,score_i_index,:]
         
        score_filtered=score_filtered[:,0:int(new_shape[1]*threshold)]
        coord_filtered=coord_filtered[:,0:int(new_shape[1]*threshold),:]
        feat_filtered=feat_filtered[:,0:int(new_shape[1]*threshold),:]

        return {'kp_score': score_filtered, 'kp_coord': coord_filtered, 'kp_feat': feat_filtered}
    

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

        kp2d_output = self.keypoint_net(input_image["color_aug", 0, 0])
        kp2d_output = self.batch_reshape_kp2d_preds(kp2d_output)

        outputs.update(kp2d_output)

        R, t = self.pose_estimator.get_pose(kp2d_output['kp_coord'], kp2d_output['kp_feat'])

        outputs["R"] = R
        outputs["t"] = t
        return outputs
