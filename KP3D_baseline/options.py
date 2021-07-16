# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
from datetime import datetime
date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class KP3DOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 #default=os.path.join(file_dir,"kitti_data")
                                 #default=os.path.join(file_dir,"../../datasets/kitti_data"))
                                 #default=os.path.join("/media/eralpkocas/hdd/TUM/AT3DCV/priordepth/MD2/", "kitti_data"))
                                 #default=os.path.join("/media/psrahul/My_Drive/my_files/Academic/TUM/Assignments/AT3DCV/PriorDepth/Git_Baseline/kitti_data/"))
                                 default="/home/ubuntu/PriorDepth/datasets/kitti_data/")

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 #default=os.path.join(os.path.expanduser("~"), "tmp"))
                                 default="/home/ubuntu/PriorDepth/KP3D_exp_logs/"+str(date_time))
                                 #default="/media/psrahul/My_Drive/my_files/Academic/TUM/Assignments/AT3DCV/PriorDepth_Phase3/kp3d_logs/"+str(date_time))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="KP3D Baseline")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_false")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
                                 #default=[0,1])

        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
                                 #default=[0, 1])
        self.parser.add_argument("--kp2d_initial_ckpt",
                                 nargs="?",
                                 type=str,
                                 help="path to the initial KP2D trained checkpoint",
                                 #default="None")
                                 default="trained_models/model_keypoint2_kitti.ckpt")
                                 #default="/media/psrahul/My_Drive/my_files/Academic/TUM/Assignments/AT3DCV/PriorDepth/Git_Baseline_2/model_keypoint2_kitti.ckpt")
                                 #default="/media/eralpkocas/hdd/TUM/AT3DCV/priordepth/KP3D_baseline/trained_models/model_keypoint2_kitti.ckpt")
        self.parser.add_argument("--depth_pretrained",
                                 help="if set, use pretrained depth network",
                                 action="store_false")
        self.parser.add_argument("--depth_encoder",
                                 nargs="?",
                                 type=str,
                                 help="path to the Monodepth encoder",
                                 default="trained_models/encoder.pth")
        self.parser.add_argument("--depth_decoder",
                                 nargs="?",
                                 type=str,
                                 help="path to the Monodepth decoder",
                                 default="trained_models/depth.pth")
        self.parser.add_argument("--visualise_images",
                                 nargs="?",
                                 type=int,
                                 help="Set to 1 to visualise images",
                                 default=0)
        self.parser.add_argument("--epipolar_distance",
                                 help="if set, use epipolar distance for threshold",
                                 action="store_false")

        self.parser.add_argument("--freeze_kp2d",
                                 nargs="?",
                                 type=int,
                                 help="Set to 0 to disable KP training",
                                 default=1)

        self.parser.add_argument("--kp_training_2dwarp",
                                 nargs="?",
                                 type=int,
                                 help="Set to 0 to disable KP training",
                                 default=0)

        self.parser.add_argument("--kp_training_2dwarp_start_epoch",
                                 nargs="?",
                                 type=int,
                                 help="Epoch to start 2D Warping Training",
                                 default=0)

        self.parser.add_argument("--kp_training_3dwarp_next",
                                 nargs="?",
                                 type=int,
                                 help="Set to 0 to disable KP training",
                                 default=0)

        self.parser.add_argument("--kp_training_3dwarp_previous",
                                 nargs="?",
                                 type=int,
                                 help="Set to 0 to disable KP training",
                                 default=0)
       
        self.parser.add_argument("--kp_training_3dwarp_start_epoch",
                                 nargs="?",
                                 type=int,
                                 help="Epoch to start 3D Warping Training",
                                 default=0)

        self.parser.add_argument("--use_pnp",
                                 help="if set, use pnp",
                                 action="store_false")

        # OPTIMIZATION options
        # TODO: in real training, check batch size but it seems like it should be 12 for md2!
        # note Konstantin: standard batch size for md is 12, correct
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8) # 8
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=10)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=5)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--with_drop",
                                 help="dropout in keypoint or not",
                                 action="store_false")
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        # TODO: in real training, adjust num workers based on the machine, in md2: 12
        # note Konstantin: our cuda machine suggests 4, i once tried it out, with 4 it was faster
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4) # 12

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=20)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")


        self.parser.add_argument("--use_posenet_for_3dwarping",
                                 type=int,
                                 help="switch warping debug mode",
                                 default=1)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
