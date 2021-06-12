import torch
import kornia


class PoseEstimation:
    def __init__(self, K1, K2):
        # TODO: check this K1&2 is correct or not!
        self.K1 = K1[:, :3, :3]
        self.K2 = K2[:, :3, :3]

    def match_keypoints(self, kp1, kp2, des1, des2):
        match_dist, match_idx = kornia.feature.match_mnn(des1, des2)
        match_kp1 = kp1[match_idx[:, 0]]
        match_kp2 = kp2[match_idx[:, 1]]
        # TODO: make this reshape more beautiful!
        match_kp1 = match_kp1.reshape((match_idx.shape[0], match_kp1.shape[0], match_kp1.shape[1]))
        match_kp2 = match_kp2.reshape((match_idx.shape[0], match_kp2.shape[0], match_kp2.shape[1]))
        return match_kp1, match_kp2

    def find_essential_matrix(self, match_kp1, match_kp2):
        # TODO: check torch.ones() part: how do we decide on matched keypoints' weights??
        
        #fun_mat = kornia.geometry.find_fundamental(match_kp1, match_kp2,
        #                                           torch.ones((match_kp1.shape[0], match_kp1.shape[1])).to("cuda"))

        ##################################DELETE THIS FOR PRODUCTION###############################################
        fun_mat = kornia.geometry.find_fundamental(match_kp1, match_kp2,
                                                   torch.ones((match_kp1.shape[0], match_kp1.shape[1])).to("cpu"))


        ess_mat = kornia.geometry.essential_from_fundamental(fun_mat, self.K1, self.K2)
        return ess_mat

    def get_six_dof(self, ess_mat, kp1, kp2):
        return kornia.geometry.motion_from_essential_choose_solution(ess_mat, self.K1, self.K2, kp1, kp2)

    def get_pose(self, kp1, kp2, des1, des2):
        match_kp1, match_kp2 = self.match_keypoints(kp1, kp2, des1, des2)
        ess_mat = self.find_essential_matrix(match_kp1, match_kp2)

        kp1 = kp1.reshape((1, kp1.shape[0], kp1.shape[1])) # TODO: make these lines more beautiful :D
        kp2 = kp2.reshape((1, kp2.shape[0], kp2.shape[1]))

        R, t, tri_points = self.get_six_dof(ess_mat, kp1, kp2)
        return R, t
