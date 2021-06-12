import time, json, os

import torch
import torch.optim as optim

from networks.kp3d_baseline import KP3D_Baseline
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
from utils2 import *
from networks.layers import *


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        # self.K = torch.tensor([[[0.58, 0, 0.5, 0],
        #                        [0, 1.92, 0.5, 0],
        #                        [0, 0, 1, 0],
        #                        [0, 0, 0, 1]]]).to("cpu" if self.opt.no_cuda else "cuda")
        self.K = torch.tensor([[[371.2000, 0.0000, 320.0000, 0.0000],
                                [0.0000, 368.6400, 96.0000, 0.0000],
                                [0.0000, 0.0000, 1.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 1.0000]]]).to("cpu" if self.opt.no_cuda else "cuda")
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        
        #this seems like a local link of yours,
        #fpath = os.path.join("/media/eralpkocas/hdd/TUM/AT3DCV/priordepth/MD2/", "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # TODO: change option.data_path for kitti_data in aws
        # konstantin note: I set the options inside md2 to "../datasets/kitti_data"
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.model = KP3D_Baseline(self.opt, self.K, self.K).to(self.device)

        self.model_optimizer = optim.Adam(self.model.parameters(), self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        print(self.opt.v1_multiscale)
        # TODO: check for multi-scale depth for md2
        print('Trainer is created successfully.')

    def train(self):
        print('in train')
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print('in run epoch')
        print("Training")
        self.model.train()
        print('line 102')
        for batch_idx, inputs in enumerate(self.train_loader):
            print('line 104')
            print(inputs.keys())
            print(len(inputs))
            before_op_time = time.time()
            print('in first batch')
            outputs, losses = self.process_batch(inputs)

            print('after process_batch')

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            print('line 114')
            self.model_lr_scheduler.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0
            print('line 122')
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                # TODO: check for depth_gt in inputs and whether we need it or not
                # TODO: if it is needed work on adding it with also check whether compute_depth_losses works for us
                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, losses)
                print('line 128')
                # TODO: as final check logging!
                self.log("train", inputs, outputs, losses)
                self.val()
                print('line 131')
            print('exiting run epoch')
            self.step += 1

    def process_batch(self, inputs):
        print('in process batch')
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        print('before forward')
        outputs = self.model(inputs)
        print('after forward')
        # TODO: generate_images_pred should produce warped images based on rotation and translation
        # TODO: make it work for sure :D
        # self.generate_images_pred(inputs, outputs)
        print(inputs)
        print(outputs)
        exit(0)
        # TODO: color and color_aug images are in inputs
        # TODO: have projected corresponding outputs for this function to calculate reprojection loss
        losses = self.compute_reprojection_loss(inputs, outputs)
        print(losses)
        exit(0)
        return outputs, losses
    # TODO: if train works correctly, this should be easy. Work on val() after being sure train works correctly.
    def val(self):
        pass

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                # if self.opt.predictive_mask:
                #     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                #         writer.add_image(
                #             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                #             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                #             self.step)

                # elif not self.opt.disable_automasking:
                #     writer.add_image(
                #         "automask_{}/{}".format(s, j),
                #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # TODO: change for all baseline model including KeypointNet!
        for model_name, model in self.model.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        # TODO: change for all baseline model including KeypointNet!
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.model[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
