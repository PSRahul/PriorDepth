import torch.nn as nn
import matplotlib.pyplot as plt
from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_estimation import PoseEstimation
from .keypoint_net import KeypointNet
from .keypoint_resnet import KeypointResnet
from  datasets.kp2d_augmentations import *
from layers import *
import torch
import os
from .pose_cnn import *
from .pose_decoder import *
            

class KP3D_Baseline(nn.Module):
    def __init__(self, options, K1, K2, epoch, batch_idx):
        super(KP3D_Baseline, self).__init__()
        self.opt = options
        self.use_pnp = self.opt.use_pnp
        if torch.cuda.is_available() and not self.opt.no_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print("LOADING MONODEPTH2 ENCODER")
        self.depth_encoder = ResnetEncoder(18, True)
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, self.opt.scales)
        #self.opt.weights_init == "pretrained"
        # extract the height and width of image that this model was trained with
        if self.opt.depth_pretrained:
            loaded_dict_enc = torch.load(self.opt.depth_encoder, map_location=device)
            feed_height = loaded_dict_enc['height']
            feed_width = loaded_dict_enc['width']
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
            self.depth_encoder.load_state_dict(filtered_dict_enc)
            self.depth_encoder.to(device)
            #self.depth_encoder.eval()

            print("LOADING MONODEPTH2 DECODER")
            loaded_dict = torch.load(self.opt.depth_decoder, map_location=device)
            self.depth_decoder.load_state_dict(loaded_dict)
            self.depth_decoder.to(device)
            #self.depth_decoder.eval()
        else:
            self.depth_encoder.to(device)
            self.depth_decoder.to(device)
        #self.keypoint_net =  KeypointResnet()
        self.keypoint_net = KeypointNet()
        
        if (self.opt.kp2d_initial_ckpt!="None"):
            print("Using pretrained Model for KP2D",self.opt.kp2d_initial_ckpt)
            checkpoint = torch.load(self.opt.kp2d_initial_ckpt, map_location=device)
            self.keypoint_net.load_state_dict(checkpoint['state_dict'])

        if self.opt.freeze_kp2d:
            for param in self.keypoint_net.parameters():
                param.requires_grad = False    


        if self.opt.use_posenet_for_3dwarping:
            print("Using PoseNet for Pose Calculations") 
            #pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
            #pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

            self.pose_encoder = ResnetEncoder(18, True, 2)
            self.pose_encoder.load_state_dict(torch.load("trained_models/pose_encoder.pth"))
            self.pose_encoder.to(device)

            self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, 1, 2)
            self.pose_decoder.load_state_dict(torch.load("trained_models/pose.pth"))
            self.pose_decoder.to(device)

            # self.pose_encoder.cuda()
            # self.pose_encoder.eval()
            # self.pose_decoder.cuda()
            # self.pose_decoder.eval()

            print("Loaded PoseNet")

        #for param in self.depth_encoder.parameters():
        #    param.requires_grad = False

        #for param in self.depth_decoder.parameters():
        #    param.requires_grad = False    

        if not os.path.exists(self.opt.log_dir+"/keypoint_vis"):
            os.makedirs(self.opt.log_dir+"/keypoint_vis")
        
        self.pose_estimator = PoseEstimation(K1, K2, self.opt.no_cuda, self.opt.log_dir,
                                             self.opt.visualise_images, self.opt.epipolar_distance)
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
        #print("epoch",epoch)

        kp2d_output1 = self.keypoint_net(input_image["color", 0, 0])
        score, coord, feat = kp2d_output1
        org_shape=score.shape
        new_shape=(org_shape[0],org_shape[2]*org_shape[3],org_shape[1])
        #print(org_shape)
        #rint(new_shape)
        #print(score.shape)
        #print(feat.shape)
        score, coord, feat = score.permute(0,2,3,1),coord.permute(0,2,3,1),feat.permute(0,2,3,1)
        if(self.training==0): 
            score, coord, feat = score.reshape((new_shape[0],new_shape[1])),coord.reshape((new_shape[0],new_shape[1],2)),feat.reshape((new_shape[0],new_shape[1],256))
        else:
            score, coord, feat = score.reshape((new_shape[0],new_shape[1])),coord.reshape((new_shape[0],new_shape[1],2)),feat.reshape((new_shape[0],new_shape[1]*4,256))
        
        kp2d_output1=[score,coord,feat]
        
        if (epoch>=self.opt.kp_training_2dwarp_start_epoch):
            if self.opt.kp_training_2dwarp:
                source_score, source_uv_pred, source_feat=self.keypoint_net(input_image["color_aug_wrapped_kp2d", 0, 0])
                target_score, target_uv_pred, target_feat=kp2d_output1
                outputs["source_score"] = source_score
                outputs["source_uv_pred"] = source_uv_pred
                outputs["source_feat"] =source_feat
                outputs["target_score"] = target_score
                outputs["target_uv_pred"] = target_uv_pred
                outputs["target_feat"] = target_feat


        #kp2d_output1 = self.batch_reshape_kp2d_preds(kp2d_output1, 1)

        kp2d_output2 = self.keypoint_net(input_image["color", 1, 0])

        score, coord, feat = kp2d_output2
        #print(coord.shape)
        #print(feat.shape)
        #print(score.shape)
        #print(coord[0,:,:4,:4])
        org_shape=score.shape
        new_shape=(org_shape[0],org_shape[2]*org_shape[3],org_shape[1])
        score, coord, feat = score.permute(0,2,3,1),coord.permute(0,2,3,1),feat.permute(0,2,3,1)
        
        if(self.training==0): 
            score, coord, feat = score.reshape((new_shape[0],new_shape[1])),coord.reshape((new_shape[0],new_shape[1],2)),feat.reshape((new_shape[0],new_shape[1],256))
        else:
            score, coord, feat = score.reshape((new_shape[0],new_shape[1])),coord.reshape((new_shape[0],new_shape[1],2)),feat.reshape((new_shape[0],new_shape[1]*4,256))
        

        kp2d_output2=[score,coord,feat]
        #print(coord[0,:16,:])
        #print(kp2d_output2[0].shape)
        #print(kp2d_output2[1].shape)
        #print(kp2d_output2[2].shape)
        
        if (epoch>=self.opt.kp_training_3dwarp_start_epoch):
            if self.opt.kp_training_3dwarp_next:
                source_score, source_uv_pred, source_feat=kp2d_output2
                outputs["source_score_next"] = source_score
                outputs["source_uv_pred_next"] = source_uv_pred
                outputs["source_feat_next"] =source_feat
            
        #kp2d_output2 = self.batch_reshape_kp2d_preds(kp2d_output2, 2)
        #print(kp2d_output2.keys())
        #print(kp2d_output2['kp2_coord'].shape)
        #print(kp2d_output2['kp2_feat'].shape)
        #rint(kp2d_output2['kp2_score'].shape)
      

        kp2d_output3 = self.keypoint_net(input_image["color", -1, 0])
        score, coord, feat = kp2d_output3
        org_shape=score.shape
        new_shape=(org_shape[0],org_shape[2]*org_shape[3],org_shape[1])

        score, coord, feat = score.permute(0,2,3,1),coord.permute(0,2,3,1),feat.permute(0,2,3,1)
        
        if(self.training==0): 
            score, coord, feat = score.reshape((new_shape[0],new_shape[1])),coord.reshape((new_shape[0],new_shape[1],2)),feat.reshape((new_shape[0],new_shape[1],256))
        else:
            score, coord, feat = score.reshape((new_shape[0],new_shape[1])),coord.reshape((new_shape[0],new_shape[1],2)),feat.reshape((new_shape[0],new_shape[1]*4,256))
        
        kp2d_output3=[score,coord,feat]
        
        if (epoch>=self.opt.kp_training_3dwarp_start_epoch):
            if self.opt.kp_training_3dwarp_previous:
                source_score, source_uv_pred, source_feat=kp2d_output3
                outputs["source_score_previous"] = source_score
                outputs["source_uv_pred_previous"] = source_uv_pred
                outputs["source_feat_previous"] =source_feat
            
        #kp2d_output3 = self.batch_reshape_kp2d_preds(kp2d_output3, 3)

        #outputs.update(kp2d_output1)
        #outputs.update(kp2d_output2)
        #outputs.update(kp2d_output3)

        if self.use_pnp:
            _, depth = disp_to_depth(disp_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            R_t1, t_t1 = self.pose_estimator.get_pose_pnp(depth, input_image["color", 0, 0], input_image["color", 1, 0],
                                                      kp2d_output1[1], kp2d_output2[1],
                                                      kp2d_output1[2], kp2d_output2[2],
                                                      epoch, batch_idx,self.training)
            R_t2, t_t2 = self.pose_estimator.get_pose_pnp(depth, input_image["color", 0, 0], input_image["color", -1, 0],
                                                      kp2d_output1[1], kp2d_output3[1],
                                                      kp2d_output1[2], kp2d_output3[2],
                                                      epoch, batch_idx,self.training)
            t_t1 = torch.unsqueeze(t_t1, dim=2)
            t_t2 = torch.unsqueeze(t_t2, dim=2)
        else:
        
        
            R_t1, t_t1 = self.pose_estimator.get_pose(input_image["color", 0, 0],input_image["color", 1, 0],
                                                kp2d_output1[1], kp2d_output2[1],
                                                kp2d_output1[2], kp2d_output2[2],
                                                epoch,batch_idx,self.training)

            #print("R_t1",R_t1.shape)
            #print("t_t1",t_t1.shape)
            R_t2, t_t2 = self.pose_estimator.get_pose(input_image["color", 0, 0],input_image["color", -1, 0],
                                                kp2d_output1[1], kp2d_output3[1],
                                                kp2d_output1[2], kp2d_output3[2],
                                                epoch,batch_idx,self.training)

        if self.opt.use_posenet_for_3dwarping:
           
            pose_feats = {f_i: input_image["color", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.pose_decoder(pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            """
                pose_inputs = torch.cat([input_image[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
                
                pose_inputs=self.pose_encoder(pose_inputs)
                axisangle, translation = self.pose_encoder(pose_inputs)
                
                for i, f_i in enumerate(self.opt.frame_ids[1:]):
                    if f_i != "s":
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])

                
                all_color_aug = torch.cat([input_image[("color_aug", i, 0)] for i in [1, 0]], 1)
                features = [self.pose_encoder(all_color_aug)]
                axisangle, translation = self.pose_decoder(features)
                print("axisangle",axisangle.shape)
                print("translation +1",translation.shape)
                pose_output=transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                #print("temp1",temp1.shape)
                outputs["pose_output_t1"] = pose_output


                all_color_aug_1 = torch.cat([input_image[("color_aug", i, 0)] for i in [-1,0]], 1)
                features = [self.pose_encoder(all_color_aug_1)]
                axisangle, translation = self.pose_decoder(features)
                #print("axisangle",axisangle.shape)
                print("translation -1",translation)
                pose_output=transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                #print("temp1",temp1.shape)
                outputs["pose_output_t2"] = pose_output
            """
        
        outputs["R_t1"] = R_t1
        outputs["t_t1"] = t_t1
        outputs["R_t2"] = R_t2
        outputs["t_t2"] = t_t2

        return outputs
