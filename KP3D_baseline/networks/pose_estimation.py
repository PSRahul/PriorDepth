import torch
import kornia
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np


class PoseEstimation:
    def __init__(self, K1, K2, cuda,log_dir):
        # TODO: check this K1&2 is correct or not!
        self.K1 = K1[:, :3, :3]
        self.K2 = K2[:, :3, :3]
        self.device = torch.device("cpu" if cuda else "cuda")
        self.log_dir=log_dir

    def match_keypoints(self, kp1, kp2, des1, des2):
        match_dist, match_idx = kornia.feature.match_mnn(des1, des2)
        match_kp1 = kp1[match_idx[:, 0]]
        match_kp2 = kp2[match_idx[:, 1]]
        match_kp1 = torch.unsqueeze(match_kp1, dim=0)
        match_kp2 = torch.unsqueeze(match_kp2, dim=0)
        return match_kp1, match_kp2

    def find_essential_matrix(self, match_kp1, match_kp2):
        fun_mat = kornia.geometry.find_fundamental(match_kp1, match_kp2,
                                                   torch.ones((match_kp1.shape[0], match_kp1.shape[1])).to(self.device))

        ess_mat = kornia.geometry.essential_from_fundamental(fun_mat, self.K1, self.K2)
        return ess_mat

    def get_six_dof(self, ess_mat, kp1, kp2):
        return kornia.geometry.motion_from_essential_choose_solution(ess_mat, self.K1, self.K2, kp1, kp2)

    def visualise_matches(self,input_image_1,input_image_2,kp1,kp2,epoch,batch_idx,batch_size):
        image_1,image_2=np.rollaxis(input_image_1[batch_size-1,:,:,:].detach().numpy(),0,3),np.rollaxis(input_image_2[batch_size-1,:,:,:].detach().numpy(),0,3)
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

        fig.savefig(self.log_dir+"keypoint_vis/epoch"+str(epoch)+"batch_idx"+str(batch_idx)+".png",bbox_inches='tight')
                  

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

            ess_mat = self.find_essential_matrix(match_kp1, match_kp2)

            R, t, tri_points = self.get_six_dof(ess_mat, match_kp1, match_kp2)
            outputs_R = torch.cat((outputs_R, R), dim=0)
            outputs_t = torch.cat((outputs_t, t), dim=0)
            #print("Checkpoint 1")
    
        if(batch_idx%250==0):
            with torch.no_grad():
                self.visualise_matches(input_image_1,input_image_2,match_kp1[0,:,:],match_kp2[0,:,:],epoch,batch_idx,batch_size)

        return outputs_R, outputs_t

