import torch.nn as nn
import matplotlib.pyplot as plt
from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_estimation import PoseEstimation
from .keypoint_net import KeypointNet
from .kp2d_losses import *
from  datasets.kp2d_augmentations import *
import torch
import os


class KP3D_Baseline(nn.Module):
    def __init__(self, options, K1, K2,epoch,batch_idx):
        super(KP3D_Baseline, self).__init__()
        self.opt = options

        if torch.cuda.is_available() and not self.opt.no_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print("LOADING MONODEPTH2 ENCODER")
        self.depth_encoder = ResnetEncoder(18, False)
        #self.opt.weights_init == "pretrained"
        # extract the height and width of image that this model was trained with

        loaded_dict_enc = torch.load(self.opt.depth_encoder, map_location=device)   
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
        self.depth_encoder.load_state_dict(filtered_dict_enc)
        self.depth_encoder.to(device)
        #self.depth_encoder.eval()

        print("LOADING MONODEPTH2 DECODER")
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, self.opt.scales)
        loaded_dict = torch.load(self.opt.depth_decoder, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(device)
        #self.depth_decoder.eval()
   
        self.keypoint_net = KeypointNet()
        
        if (self.opt.kp2d_initial_ckpt!="None"):
            print("Using pretrained Model for KP2D",self.opt.kp2d_initial_ckpt)
            checkpoint = torch.load(self.opt.kp2d_initial_ckpt, map_location=device)
            self.keypoint_net.load_state_dict(checkpoint['state_dict'])

        for param in self.keypoint_net.parameters():
            param.requires_grad = False    

        
        #for param in self.depth_encoder.parameters():
        #    param.requires_grad = False

        #for param in self.depth_decoder.parameters():
        #    param.requires_grad = False    

        if not os.path.exists(self.opt.log_dir+"/keypoint_vis"):
            os.makedirs(self.opt.log_dir+"/keypoint_vis")
        
        self.pose_estimator = PoseEstimation(K1, K2, self.opt.no_cuda,self.opt.log_dir,self.opt.visualise_images)
        ## TODO: // add K1 and K2 to options! or check whether K is correct in trainer.py line 36

   
    def batch_reshape_kp2d_preds(self, kp2d_output, j, threshold=0.3):
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
         
        score_filtered=score_filtered[:,0:int(new_shape[1]*threshold)].to("cpu" if self.opt.no_cuda else "cuda")
        coord_filtered=coord_filtered[:,0:int(new_shape[1]*threshold),:].to("cpu" if self.opt.no_cuda else "cuda")
        feat_filtered=feat_filtered[:,0:int(new_shape[1]*threshold),:].to("cpu" if self.opt.no_cuda else "cuda")

        return {'kp{}_score'.format(j): score_filtered, 'kp{}_coord'.format(j): coord_filtered, 'kp{}_feat'.format(j): feat_filtered}
    

    def forward(self, input_image,epoch,batch_idx):
        # make sure color, 0, 0 is target image and color, 1, 0 is context image
        # calculate transformation matrix from target to context
        # warp target pixels to obtain context pixels
        # compute photometric loss between target image and warped target image
        outputs = {}

        depth_features = self.depth_encoder(input_image["color_aug", 0, 0])
        disp_outputs = self.depth_decoder(depth_features)
        outputs.update(disp_outputs)

        #print(input_image["color_aug", 0, 0].shape)
        #print(input_image["color_aug", 1, 0].shape)
        #print(input_image["color_aug", -1, 0].shape)

        #target_img,source_img,homography=ha_augment_sample(input_image["color_aug", 0, 0][0,:,:,:])
        #print("Target Shape",target_img.shape)  
        #print("Souce Shape",source_img.shape)
        #print("Homography Shape",homography.shape)


        #plt.imsave("target_img.png",target_img.permute(1,2,0).detach().cpu().numpy())
        #plt.imsave("source_img.png",source_img.permute(1,2,0).detach().cpu().numpy())
        #plt.imsave("input.png",input_image["color_aug", 0, 0][0,:,:,:].permute(1,2,0).detach().cpu().numpy())
        #plt.imsave("input_wrapped.png",input_image["color_aug_wrapped_kp2d", 0, 0][0,:,:,:].permute(1,2,0).detach().cpu().numpy())


        kp2d_output1 = self.keypoint_net(input_image["color_aug", 0, 0])
        kp2d_output1 = self.batch_reshape_kp2d_preds(kp2d_output1, 1)

        kp2d_output2 = self.keypoint_net(input_image["color_aug", 1, 0])
        kp2d_output2 = self.batch_reshape_kp2d_preds(kp2d_output2, 2)

        kp2d_output3 = self.keypoint_net(input_image["color_aug", -1, 0])
        kp2d_output3 = self.batch_reshape_kp2d_preds(kp2d_output3, 3)

        outputs.update(kp2d_output1)
        outputs.update(kp2d_output2)
        outputs.update(kp2d_output3)

        R_t1, t_t1 = self.pose_estimator.get_pose(input_image["color_aug", 0, 0],input_image["color_aug", 1, 0],
                                            kp2d_output1['kp1_coord'], kp2d_output2['kp2_coord'],
                                            kp2d_output1['kp1_feat'], kp2d_output2['kp2_feat'],
                                            epoch,batch_idx)

        R_t2, t_t2 = self.pose_estimator.get_pose(input_image["color_aug", 0, 0],input_image["color_aug", -1, 0],
                                            kp2d_output1['kp1_coord'], kp2d_output3['kp3_coord'],
                                            kp2d_output1['kp1_feat'], kp2d_output3['kp3_feat'],
                                            epoch,batch_idx)

        outputs["R_t1"] = R_t1
        outputs["t_t1"] = t_t1
        outputs["R_t2"] = R_t2
        outputs["t_t2"] = t_t2

        return outputs
