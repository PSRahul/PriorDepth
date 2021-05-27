import torch
import kornia


class EstimatePose():
    def __init__(self, K1, K2, kp1, des1, kp2, des2):
        self.K1 = K1
        self.K2 = K2
        self.kp1 = kp1
        self.des1 = des1
        self.kp2 = kp2
        self.des2 = des2

    def match_keypoints(self, des1, des2):
        match_dist, match_idx = kornia.feature.match_mnn(des1, des2)
        match_kp1 = self.kp1[match_idx[:, 0]]
        match_kp2 = self.kp2[match_idx[:, 1]]
        return match_kp1, match_kp2

    def find_essential_matrix(self, match_kp1, match_kp2):
        fun_mat = kornia.geometry.find_fundamental(match_kp1, match_kp2,
                                                   torch.ones((match_kp1.shape[0]. match_kp1.shape[1])))
        ess_mat = kornia.geometry.essential_from_fundamental(fun_mat, self.K1, self.K2)
        return ess_mat

    def get_six_dof(self, ess_mat):
        return kornia.geometry.motion_from_essential_choose_solution(ess_mat, self.K1, self.K2, self.kp1, self.kp2)

    def get_pose(self):
        match_kp1, match_kp2 = self.match_keypoints(self.des1, self.des2)
        ess_mat = self.find_essential_matrix(match_kp1, match_kp2)
        R, t, tri_points = self.get_six_dof(ess_mat)
        return R, t
