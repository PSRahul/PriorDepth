import torch
import kornia
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.ops import perspective_n_points
import sys

class PoseEstimation:
    def __init__(self, K1, K2, cuda,log_dir,visualise_images, epipolar_dist):
        # TODO: check this K1&2 is correct or not!
        self.K1 = K1[:, :3, :3]
        self.K2 = K2[:, :3, :3]
        # print('=========================')
        # print('K1 and K2 in initialization of Pose Estimation')
        # print('K1', K1)
        # print('K2', K2)
        # print('=========================')
        #self.K1[:, 0, :] *= 640
        #self.K1[:, 1, :] *= 192
        #self.K2[:, 0, :] *= 640
        #self.K2[:, 1, :] *= 192
        # print('=========================')
        # print('self K1', self.K1)
        # print('self K2', self.K2)
        # print('=========================')
        self.device = torch.device("cpu" if cuda else "cuda")
        self.log_dir=log_dir
        self.visualise_images=visualise_images
        self.epipolar_distance = epipolar_dist

    def good_matches(self, match_dist, match_idx, threshold=150):
        good_matches = []
        good_match_idx = []
        len_matches = match_dist.shape[0]
        for i in range(len_matches):
            dist = match_dist[i]
            if dist < threshold:
                good_matches.append(dist)
                good_match_idx.append(match_idx[i].cpu().numpy())

        good_matches = torch.tensor(good_matches).to(self.device)
        good_match_idx = torch.tensor(good_match_idx).to(self.device)

        return good_matches, good_match_idx

    def match_keypoints(self, kp1, kp2, des1, des2):
        match_dist, match_idx = kornia.feature.match_mnn(des1, des2)
        match_dist, match_idx = self.good_matches(match_dist, match_idx)

        match_kp1 = kp1[match_idx[:, 0]]
        match_kp2 = kp2[match_idx[:, 1]]

        match_kp1 = torch.unsqueeze(match_kp1, dim=0)
        match_kp2 = torch.unsqueeze(match_kp2, dim=0)
        return match_kp1, match_kp2

    def find_essential_matrix(self, match_kp1, match_kp2):
        fun_mat = kornia.geometry.find_fundamental(match_kp1, match_kp2,
                                                   torch.ones((match_kp1.shape[0], match_kp1.shape[1])).to(self.device))

        ess_mat = kornia.geometry.essential_from_fundamental(fun_mat, self.K1, self.K2)
        return ess_mat, fun_mat

    def find_essential_matrix_batch(self, match_kp1_batch, match_kp2_batch,match_weights):
        fun_mat_batch = kornia.geometry.find_fundamental(match_kp1_batch, match_kp2_batch,
                                                   match_weights)

        ess_mat_batch = kornia.geometry.essential_from_fundamental(fun_mat_batch, self.K1, self.K2)
        return ess_mat_batch, fun_mat_batch

    def get_six_dof(self, ess_mat, kp1, kp2):
        return kornia.geometry.motion_from_essential_choose_solution(ess_mat, self.K1, self.K2, kp1, kp2)

    def visualise_matches(self,input_image_1,input_image_2,kp1,kp2,epoch,batch_idx,batch_size):
        image_1,image_2=np.rollaxis(input_image_1[batch_size-1,:,:,:].cpu().detach().numpy(),0,3),np.rollaxis(input_image_2[batch_size-1,:,:,:].cpu().detach().numpy(),0,3)
        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        #fig.suptitle('Horizontally stacked subplots')
        _ =ax1.imshow(image_1)
        _ =ax1.scatter(kp1[:,0],kp1[:,1],s=0.6,c="r",marker="X")
        _ =ax2.imshow(image_2)
        _ =ax2.scatter(kp2[:,0],kp2[:,1],s=0.6,c="cyan",marker="X")
        ax1.axis('off')
        ax2.axis("off")

        for i in range(30):#kp1.shape[0]):
            xy_a = (kp1[i,0],kp1[i,1])
            xy_b = (kp2[i,0],kp2[i,1])
            con = ConnectionPatch(xyA=xy_a, xyB=xy_b, coordsA="data", coordsB="data",axesA=ax2, axesB=ax1, color="lime")
            ax2.add_artist(con)

        fig.savefig(self.log_dir+"/keypoint_vis/epoch"+str(epoch)+"batch_idx"+str(batch_idx)+".png",bbox_inches='tight')
                  

    def get_pose(self, input_image_1,input_image_2,kp1, kp2, des1, des2,epoch,batch_idx):
        outputs_R = torch.tensor([]).to(self.device)
        outputs_t = torch.tensor([]).to(self.device)
        batch_size=kp1.shape[0]
        for i in range(kp1.shape[0]):
            curr_kp1 = kp1[i, :, :]
            curr_kp2 = kp2[i, :, :]
            curr_des1 = des1[i, :, :]
            curr_des2 = des2[i, :, :]

            match_kp1, match_kp2 = self.match_keypoints(curr_kp1, curr_kp2, curr_des1, curr_des2)
            ess_mat, fun_mat = self.find_essential_matrix(match_kp1, match_kp2)
            if self.epipolar_distance:
                distances = kornia.geometry.symmetrical_epipolar_distance(match_kp1, match_kp2, fun_mat)
                mask = distances < 0.03
                mask = torch.stack((mask[0, :], mask[0, :]), dim=1)
                mask = torch.unsqueeze(mask, dim=0)
                match_kp1 = torch.masked_select(match_kp1[0], mask)
                match_kp2 = torch.masked_select(match_kp2[0], mask)
                match_kp1 = torch.reshape(match_kp1, (1, int(match_kp1.shape[0] / 2), 2))
                match_kp2 = torch.reshape(match_kp2, (1, int(match_kp2.shape[0] / 2), 2))
            R, t, tri_points = self.get_six_dof(ess_mat, match_kp1, match_kp2)
            outputs_R = torch.cat((outputs_R, R), dim=0)
            outputs_t = torch.cat((outputs_t, t), dim=0)

        if(self.visualise_images):
            if(batch_idx%250==0):
                with torch.no_grad():
                    self.visualise_matches(input_image_1,input_image_2,match_kp1[0,:,:].cpu(),match_kp2[0,:,:].cpu(),epoch,batch_idx,batch_size)
            plt.close('all')
        return outputs_R, outputs_t


    def get_pose_batch(self, input_image_1,input_image_2,kp1, kp2, des1, des2,epoch,batch_idx):
        outputs_R = torch.tensor([]).to(self.device)
        outputs_t = torch.tensor([]).to(self.device)
        batch_size=kp1.shape[0]
        match_kp1_batch=torch.zeros((batch_size,600,2),device=self.device)
        match_kp2_batch=torch.zeros((batch_size,600,2),device=self.device)
        match_weights=torch.zeros((batch_size,600),device=self.device)

        for i in range(kp1.shape[0]):
            curr_kp1 = kp1[i, :, :]
            curr_kp2 = kp2[i, :, :]
            curr_des1 = des1[i, :, :]
            curr_des2 = des2[i, :, :]

            match_kp1, match_kp2 = self.match_keypoints(curr_kp1, curr_kp2, curr_des1, curr_des2)
            ess_mat, fun_mat = self.find_essential_matrix(match_kp1, match_kp2)

            match_kp1_batch[i,:match_kp1.shape[1],:]=match_kp1[0,:,:]
            match_kp2_batch[i,:match_kp2.shape[1],:]=match_kp2[0,:,:]
            match_weights[i,:match_kp1.shape[1]]=1
          
        ess_mat_batch, fun_mat_batch = self.find_essential_matrix_batch(match_kp1_batch, match_kp2_batch,match_weights)

        if self.epipolar_distance:
            distances = kornia.geometry.symmetrical_epipolar_distance(match_kp1_batch, match_kp2_batch, fun_mat_batch)
            mask = distances < 0.03
            mask_org=mask
            #print("Masks shape",mask.shape)
            mask = torch.stack((mask, mask), dim=-1)
            match_kp1_batch[mask==False]=0
            match_kp2_batch[mask==False]=0
            #print("KP Shape",match_kp1_batch.shape)

            outputs_R, outputs_t, tri_points =kornia.geometry.motion_from_essential_choose_solution(ess_mat_batch, self.K1, self.K2,
                                                     match_kp1_batch, match_kp2_batch,mask_org)
                
        if(self.visualise_images):
            if(batch_idx%250==0):
                with torch.no_grad():
                    self.visualise_matches(input_image_1,input_image_2,match_kp1[0,:,:].cpu(),match_kp2[0,:,:].cpu(),epoch,batch_idx,batch_size)
            plt.close('all')

        return outputs_R, outputs_t

    def reproject_points(self, depth_img, kp):
        rounded_kp = np.array(np.round(kp[0].cpu().detach().numpy())).astype(int)
        depth_vals = depth_img[0, rounded_kp[:, 1], rounded_kp[:, 0]]
        kp=kp[0]
        depth_vals2 = torch.stack((depth_vals, depth_vals), dim=1)
        c_x_y = torch.Tensor([self.K1[0, 0, 2], self.K1[0, 1, 2]]).to(self.device)
        f_x_Y = torch.Tensor([self.K1[0, 0, 0], self.K1[0, 1, 1]]).to(self.device)
        kp3d = (kp - c_x_y) * depth_vals2 / f_x_Y
        kp3d = torch.cat((kp3d, torch.unsqueeze(depth_vals, dim=1)), dim=1)
        return torch.unsqueeze(kp3d, dim=0)

    def get_pose_pnp(self, depth_img, input_image_1, input_image_2, kp1, kp2, des1, des2,epoch, batch_idx):
        outputs_R = torch.tensor([]).to(self.device)
        outputs_t = torch.tensor([]).to(self.device)
        batch_size = kp1.shape[0]
        for i in range(kp1.shape[0]):
            curr_kp1 = kp1[i, :, :]
            curr_kp2 = kp2[i, :, :]
            curr_des1 = des1[i, :, :]
            curr_des2 = des2[i, :, :]
            curr_depth = depth_img[i, :, :, :]
            match_kp1, match_kp2 = self.match_keypoints(curr_kp1, curr_kp2, curr_des1, curr_des2)

            ess_mat, fun_mat = self.find_essential_matrix(match_kp1, match_kp2)
            if self.epipolar_distance:
                distances = kornia.geometry.symmetrical_epipolar_distance(match_kp1, match_kp2, fun_mat)
                mask = distances < 0.1
                mask = torch.stack((mask[0, :], mask[0, :]), dim=1)
                mask = torch.unsqueeze(mask, dim=0)
                match_kp1 = torch.masked_select(match_kp1[0], mask)
                match_kp2 = torch.masked_select(match_kp2[0], mask)
                match_kp1 = torch.reshape(match_kp1, (1, int(match_kp1.shape[0] / 2), 2))
                match_kp2 = torch.reshape(match_kp2, (1, int(match_kp2.shape[0] / 2), 2))

            kp3d = self.reproject_points(curr_depth, match_kp1)
            output = perspective_n_points.efficient_pnp(kp3d, match_kp1)

            R = output[1]
            t = output[2]
            outputs_R = torch.cat((outputs_R, R), dim=0)
            outputs_t = torch.cat((outputs_t, t), dim=0)

        if (self.visualise_images):
            if (batch_idx % 250 == 0):
                with torch.no_grad():
                    self.visualise_matches(input_image_1, input_image_2, match_kp1[0, :, :].cpu(),
                                           match_kp2[0, :, :].cpu(), epoch, batch_idx, batch_size)
            plt.close('all')
        return outputs_R, outputs_t