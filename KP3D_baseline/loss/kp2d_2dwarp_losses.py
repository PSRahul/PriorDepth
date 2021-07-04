import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

from utils.image import (image_grid, to_color_normalized,
                              to_gray_normalized)

keypoint_loss_weight=1.0
descriptor_loss_weight=2.0
score_loss_weight=1.0
descriptor_loss=True
relax_field = 4


def build_descriptor_loss(source_des, target_des, source_points, tar_points, tar_points_un, keypoint_mask=None, relax_field=8, eval_only=False):
    """Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf..
    Parameters
    ----------
    source_des: torch.Tensor (B,256,H/8,W/8)
        Source image descriptors.
    target_des: torch.Tensor (B,256,H/8,W/8)
        Target image descriptors.
    source_points: torch.Tensor (B,H/8,W/8,2)
        Source image keypoints
    tar_points: torch.Tensor (B,H/8,W/8,2)
        Target image keypoints
    tar_points_un: torch.Tensor (B,2,H/8,W/8)
        Target image keypoints unnormalized
    eval_only: bool
        Computes only recall without the loss.
    Returns
    -------
    loss: torch.Tensor
        Descriptor loss.
    recall: torch.Tensor
        Descriptor match recall.
    """
    device = source_des.device
    batch_size, C, _, _ = source_des.shape
    loss, recall = 0., 0.
    margins = 0.2

    for cur_ind in range(batch_size):

        if keypoint_mask is None:
            ref_desc = torch.nn.functional.grid_sample(source_des[cur_ind].unsqueeze(0), source_points[cur_ind].unsqueeze(0), align_corners=True).squeeze().view(C, -1)
            tar_desc = torch.nn.functional.grid_sample(target_des[cur_ind].unsqueeze(0), tar_points[cur_ind].unsqueeze(0), align_corners=True).squeeze().view(C, -1)
            tar_points_raw = tar_points_un[cur_ind].view(2, -1)
        else:
            keypoint_mask_ind = keypoint_mask[cur_ind].squeeze()

            n_feat = keypoint_mask_ind.sum().item()
            if n_feat < 20:
                continue

            ref_desc = torch.nn.functional.grid_sample(source_des[cur_ind].unsqueeze(0), source_points[cur_ind].unsqueeze(0), align_corners=True).squeeze()[:, keypoint_mask_ind]
            tar_desc = torch.nn.functional.grid_sample(target_des[cur_ind].unsqueeze(0), tar_points[cur_ind].unsqueeze(0), align_corners=True).squeeze()[:, keypoint_mask_ind]
            tar_points_raw = tar_points_un[cur_ind][:, keypoint_mask_ind]

        # Compute dense descriptor distance matrix and find nearest neighbor
        ref_desc = ref_desc.div(torch.norm(ref_desc, p=2, dim=0))
        tar_desc = tar_desc.div(torch.norm(tar_desc, p=2, dim=0))
        dmat = torch.mm(ref_desc.t(), tar_desc)
        dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, min=-1, max=1))

        # Sort distance matrix
        dmat_sorted, idx = torch.sort(dmat, dim=1)

        # Compute triplet loss and recall
        candidates = idx.t() # Candidates, sorted by descriptor distance

        # Get corresponding keypoint positions for each candidate descriptor
        match_k_x = tar_points_raw[0, candidates]
        match_k_y = tar_points_raw[1, candidates]

        # True keypoint coordinates
        true_x = tar_points_raw[0]
        true_y = tar_points_raw[1]

        # Compute recall as the number of correct matches, i.e. the first match is the correct one
        correct_matches = (abs(match_k_x[0]-true_x) == 0) & (abs(match_k_y[0]-true_y) == 0)
        recall += float(1.0 / batch_size) * (float(correct_matches.float().sum()) / float( ref_desc.size(1)))

        if eval_only:
            continue

        # Compute correct matches, allowing for a few pixels tolerance (i.e. relax_field)
        correct_idx = (abs(match_k_x - true_x) <= relax_field) & (abs(match_k_y - true_y) <= relax_field)
        # Get hardest negative example as an incorrect match and with the smallest descriptor distance 
        incorrect_first = dmat_sorted.t()
        incorrect_first[correct_idx] = 2.0 # largest distance is at most 2
        incorrect_first = torch.argmin(incorrect_first, dim=0)
        incorrect_first_index = candidates.gather(0, incorrect_first.unsqueeze(0)).squeeze()

        anchor_var = ref_desc
        pos_var    = tar_desc
        neg_var    = tar_desc[:, incorrect_first_index]

        loss += float(1.0 / batch_size) * torch.nn.functional.triplet_margin_loss(anchor_var.t(), pos_var.t(), neg_var.t(), margin=margins)

    return loss, recall


def warp_homography_batch(sources, homographies):
    """Batch warp keypoints given homographies.

    Parameters
    ----------
    sources: torch.Tensor (B,H,W,C)
        Keypoints vector.
    homographies: torch.Tensor (B,3,3)
        Homographies.

    Returns
    -------
    warped_sources: torch.Tensor (B,H,W,C)
        Warped keypoints vector.
    """
    B, H, W, _ = sources.shape
    warped_sources = []
    for b in range(B):
        source = sources[b].clone()
        source = source.view(-1,2)
        source = torch.addmm(homographies[b,:,2], source, homographies[b,:,:2].t())
        source.mul_(1/source[:,2].unsqueeze(1))
        source = source[:,:2].contiguous().view(H,W,2)
        warped_sources.append(source)
    return torch.stack(warped_sources, dim=0)


def calculate_2d_warping_loss(inputs,outputs):
  
    loss_2d = 0
    B, _, H, W = inputs[('color_aug_wrapped_kp2d', 0, 0)].shape
    device = inputs[('color_aug_wrapped_kp2d', 0, 0)].device
    homography=inputs[('homography', 0, 0)].to(device)

    source_score=outputs["source_score"] 
    source_uv_pred=outputs["source_uv_pred"] 
    source_feat=outputs["source_feat"]
    target_score=outputs["target_score"] 
    target_uv_pred=outputs["target_uv_pred"]
    target_feat=outputs["target_feat"] 
    _, _, Hc, Wc = target_score.shape


    target_uv_norm = target_uv_pred.clone()
    target_uv_norm[:,0] = (target_uv_norm[:,0] / (float(W-1)/2.)) - 1.
    target_uv_norm[:,1] = (target_uv_norm[:,1] / (float(H-1)/2.)) - 1.
    target_uv_norm = target_uv_norm.permute(0, 2, 3, 1)

    source_uv_norm = source_uv_pred.clone()
    source_uv_norm[:,0] = (source_uv_norm[:,0] / (float(W-1)/2.)) - 1.
    source_uv_norm[:,1] = (source_uv_norm[:,1] / (float(H-1)/2.)) - 1.
    source_uv_norm = source_uv_norm.permute(0, 2, 3, 1)

    source_uv_warped_norm = warp_homography_batch(source_uv_norm, homography)
    source_uv_warped = source_uv_warped_norm.clone()

    source_uv_warped[:,:,:,0] = (source_uv_warped[:,:,:,0] + 1) * (float(W-1)/2.)
    source_uv_warped[:,:,:,1] = (source_uv_warped[:,:,:,1] + 1) * (float(H-1)/2.)
    source_uv_warped = source_uv_warped.permute(0, 3, 1, 2)

    target_uv_resampled = torch.nn.functional.grid_sample(target_uv_pred, source_uv_warped_norm, mode='nearest', align_corners=True)

    target_uv_resampled_norm = target_uv_resampled.clone()
    target_uv_resampled_norm[:,0] = (target_uv_resampled_norm[:,0] / (float(W-1)/2.)) - 1.
    target_uv_resampled_norm[:,1] = (target_uv_resampled_norm[:,1] / (float(H-1)/2.)) - 1.
    target_uv_resampled_norm = target_uv_resampled_norm.permute(0, 2, 3, 1)

    # Border mask
    border_mask_ori = torch.ones(B,Hc,Wc)
    border_mask_ori[:,0] = 0
    border_mask_ori[:,Hc-1] = 0
    border_mask_ori[:,:,0] = 0
    border_mask_ori[:,:,Wc-1] = 0
    border_mask_ori = border_mask_ori.gt(1e-3).to(device)

    # Out-of-bourder(OOB) mask. Not nessesary in our case, since it's prevented at HA procedure already. Kept here for future usage.
    oob_mask2 = source_uv_warped_norm[:,:,:,0].lt(1) & source_uv_warped_norm[:,:,:,0].gt(-1) & source_uv_warped_norm[:,:,:,1].lt(1) & source_uv_warped_norm[:,:,:,1].gt(-1)
    border_mask = border_mask_ori & oob_mask2

    d_uv_mat_abs = torch.abs(source_uv_warped.view(B,2,-1).unsqueeze(3) - target_uv_pred.view(B,2,-1).unsqueeze(2))
    d_uv_l2_mat = torch.norm(d_uv_mat_abs, p=2, dim=1)
    d_uv_l2_min, d_uv_l2_min_index = d_uv_l2_mat.min(dim=2)

    dist_norm_valid_mask = d_uv_l2_min.lt(4) & border_mask.view(B,Hc*Wc)

    # Keypoint loss
    loc_loss = d_uv_l2_min[dist_norm_valid_mask].mean()
    loss_2d += keypoint_loss_weight * loc_loss.mean()

    #Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf.
    if descriptor_loss:
        metric_loss, recall_2d = build_descriptor_loss(source_feat, target_feat, source_uv_norm.detach(), source_uv_warped_norm.detach(), source_uv_warped, keypoint_mask=border_mask, relax_field=relax_field)
        loss_2d += descriptor_loss_weight * metric_loss * 2
    else:
        _, recall_2d = build_descriptor_loss(source_feat, target_feat, source_uv_norm, source_uv_warped_norm, source_uv_warped, keypoint_mask=border_mask, relax_field=relax_field, eval_only=True)

    #Score Head Loss
    target_score_associated = target_score.view(B,Hc*Wc).gather(1, d_uv_l2_min_index).view(B,Hc,Wc).unsqueeze(1)
    dist_norm_valid_mask = dist_norm_valid_mask.view(B,Hc,Wc).unsqueeze(1) & border_mask.unsqueeze(1)
    d_uv_l2_min = d_uv_l2_min.view(B,Hc,Wc).unsqueeze(1)
    loc_err = d_uv_l2_min[dist_norm_valid_mask]

    usp_loss = (target_score_associated[dist_norm_valid_mask] + source_score[dist_norm_valid_mask]) * (loc_err - loc_err.mean())
    loss_2d += score_loss_weight * usp_loss.mean()

    target_score_resampled = torch.nn.functional.grid_sample(target_score, source_uv_warped_norm.detach(), mode='bilinear', align_corners=True)

    loss_2d += score_loss_weight * torch.nn.functional.mse_loss(target_score_resampled[border_mask.unsqueeze(1)],
                                                                        source_score[border_mask.unsqueeze(1)]).mean() * 2
    
    return loss_2d