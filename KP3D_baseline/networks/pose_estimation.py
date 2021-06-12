import torch
import kornia


class PoseEstimation:
    def __init__(self, K1, K2, cuda):
        # TODO: check this K1&2 is correct or not!
        self.K1 = K1[:, :3, :3]
        self.K2 = K2[:, :3, :3]
        self.device = torch.device("cpu" if cuda else "cuda")

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

    def get_pose(self, kp1, kp2, des1, des2):
        kp1 = torch.unsqueeze(kp1, dim=0)
        kp2 = torch.unsqueeze(kp2, dim=0)
        des1 = torch.unsqueeze(des1, dim=0)
        des2 = torch.unsqueeze(des2, dim=0)

        outputs_R = torch.tensor([]).to(self.device)
        outputs_t = torch.tensor([]).to(self.device)
        for i in range(kp1.shape[0]):
            kp1 = kp1[i, :, :]
            kp2 = kp2[i, :, :]
            des1 = des1[i, :, :]
            des2 = des2[i, :, :]

            match_kp1, match_kp2 = self.match_keypoints(kp1, kp2, des1, des2)
            ess_mat = self.find_essential_matrix(match_kp1, match_kp2)

            R, t, tri_points = self.get_six_dof(ess_mat, match_kp1, match_kp2)
            outputs_R = torch.cat((outputs_R, R), dim=0)
            outputs_t = torch.cat((outputs_t, t), dim=0)
        return outputs_R, outputs_t
