import logging
import os
import time
import tqdm

import SimpleITK as sitk
import numpy as np
# from utils.utils.contour_eval import *
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from agents.base import BaseAgent
from graphs.losses.loss import *
from graphs.models.regnet import RegNet, SpatialTransformer
from graphs.models.segnet import SegNet
from utils import dataset_niftynet as dset_utils
from utils.libs import resize_image_mlvl
from utils.model_util import count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class stlAgent(BaseAgent):
    def __init__(self, args, data_config):
        super(stlAgent).__init__()

        self.args = args
        self.data_config = data_config
        self.logger = logging.getLogger()
        self.current_epoch = 0

        if self.args.mode == 'eval':
            pass
        else:

            # initialize tensorboard writer
            self.summary_writer = SummaryWriter(self.args.tensorboard_dir)
            # Create an instance from the data loader
            self.dsets = dset_utils.get_datasets(self.args, self.data_config)
            self.dataloaders = {x: DataLoader(self.dsets[x], batch_size=self.args.batch_size,
                                              shuffle=True, num_workers=self.args.num_threads)
                                for x in self.args.split_set}
            # Create an instance from the Model
            if self.args.network == 'Seg':
                self.model = SegNet(classes=self.args.num_classes_seg, depth=3, initial_channels=8,
                                    channels_list = self.args.num_featurmaps).to(device)
                # Create instance from the loss
                self.dice_loss = multi_dice_loss().to(device)
            elif self.args.network == 'Reg':
                self.model = RegNet(dim=3, scale=self.args.scale,
                                    depth=3, initial_channels=8,
                                    normalization='batchNorm').to(device)
                # Create instance from the loss
                self.ncc_loss = NCC(self.args.dim, self.args.ncc_window_size).to(device)
                self.spatial_transform = SpatialTransformer(dim=self.args.dim)
            else:
                print('Unknown Netowrk')

            self.logger.info(self.model)
            self.logger.info(f"Total Trainable Params: {count_parameters(self.model)}")

            # Create instance from the optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
            # Model Loading from the latest checkpoint if not found start from scratch.
            self.load_checkpoint()

    def save_checkpoint(self, filename='model.pth.tar', is_best=False):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        # Save the state
        torch.save(state, os.path.join(self.args.model_dir, filename))

    def load_checkpoint(self, filename='model.pth.tar'):
        filename = os.path.join(self.args.model_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=device)

            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Model loaded successfully from '{}' at (epoch {}) \n"
                             .format(self.args.model_dir, checkpoint['epoch']))
        except OSError as e:
            self.logger.info("No model exists from '{}'. Skipping...".format(self.args.model_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.args.mode == 'train':
                self.train()
            elif self.args.mode == 'inference':
                self.inference()
            elif self.args.mode == 'eval':
                self.eval()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        since = time.time()

        for epoch in range(self.current_epoch, self.args.num_epochs):

            self.logger.info('-' * 10)
            self.logger.info('Epoch {}/{}'.format(epoch, self.args.num_epochs))

            self.current_epoch = epoch
            self.train_one_epoch()

            if (epoch) % self.args.validate_every == 0:
                self.validate()

            self.save_checkpoint()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def train_one_epoch(self):

        # Set model to training mode
        self.model.train()
        # initialize stats
        running_total_loss = 0.
        running_simi_loss = 0.
        running_smooth_loss = 0.
        running_total_seg_dsc_loss = 0.
        running_total_reg_dsc_loss = 0.
        running_total_focal_loss = 0.

        running_seg_dsc_LV = 0.
        running_seg_dsc_LVM = 0.
        running_seg_dsc_RV = 0.
        running_seg_dsc_LA = 0.
        running_seg_dsc_RA = 0.

        running_reg_dsc_LV = 0.
        running_reg_dsc_LVM = 0.
        running_reg_dsc_RV = 0.
        running_reg_dsc_LA = 0.
        running_reg_dsc_RA = 0.
        running_alpha_enc = {}
        running_alpha_dec = {}

        for batch_idx, (fimage, flabel, mimage, mlabel) in enumerate(self.dataloaders['inferecne'], 1):
            # switch model to training mode, clear gradient accumulators
            self.model.train()
            self.optimizer.zero_grad()
            total_loss = 0.

            # format the input images in the right format for pytorch model
            nbatches, wsize, nchannels, x, y, z, _ = fimage.size()
            fimage = fimage.view(nbatches * wsize, nchannels, x, y, z).to(self.args.device)  #(n, 1, d, w, h)
            flabel = flabel.view(nbatches * wsize, nchannels, x, y, z).to(self.args.device)
            mimage = mimage.view(nbatches * wsize, nchannels, x, y, z).to(self.args.device)
            mlabel = mlabel.view(nbatches * wsize, nchannels, x, y, z).to(self.args.device)

            # normalize image intensity
            fimage[fimage > 1000] = 1000
            fimage[fimage < -1000] = -1000
            fimage = fimage / 1000

            mimage[mimage > 1000] = 1000
            mimage[mimage < -1000] = -1000
            mimage = mimage / 1000

            # resize the images for different resolutions
            flabel_high = resize_image_mlvl(self.args, flabel, 0)
            flabel_mid = resize_image_mlvl(self.args, flabel, 1)
            flabel_low = resize_image_mlvl(self.args, flabel, 2)

            fimage_high = resize_image_mlvl(self.args, fimage, 0)
            fimage_mid = resize_image_mlvl(self.args, fimage, 1)
            fimage_low = resize_image_mlvl(self.args, fimage, 2)

            mlabel_high = resize_image_mlvl(self.args, mlabel, 0)
            mlabel_mid = resize_image_mlvl(self.args, mlabel, 1)
            mlabel_low = resize_image_mlvl(self.args, mlabel, 2)

            mimage_high = resize_image_mlvl(self.args, mimage, 0)
            mimage_mid = resize_image_mlvl(self.args, mimage, 1)
            mimage_low = resize_image_mlvl(self.args, mimage, 2)

            res = self.model(batch_mri)

            if 'probs' in res: # segmentation task

                mean_seg_dsc_list = self.dice_loss(batch_label, res['probs'], 'seg')
                mean_seg_dsc_loss_item = mean_seg_dsc_list.mean().item()
                total_loss += mean_seg_dsc_list.mean()

                if self.args.focal_weight > 0:
                    mean_focal_list = self.focal_loss(batch_label, res['probs'])
                    mean_focal_loss_item = mean_focal_list.mean().item()
                    total_loss += self.args.focal_weight * mean_focal_list.mean()
                else:
                    mean_focal_loss_item = 0

                running_total_focal_loss += mean_focal_loss_item
                running_total_seg_dsc_loss += mean_seg_dsc_loss_item
                running_seg_dsc_LV += 1.0 - mean_seg_dsc_list[0].item()
                running_seg_dsc_LVM += 1.0 - mean_seg_dsc_list[1].item()
                running_seg_dsc_RV += 1.0 - mean_seg_dsc_list[2].item()
                running_seg_dsc_LA += 1.0 - mean_seg_dsc_list[3].item()
                running_seg_dsc_RA += 1.0 - mean_seg_dsc_list[4].item()

            if 'disp_t2i' in res: # if registration task or multi-task

                if 'disp_i2t' in res:
                    simi_loss = (self.ncc_loss(res['warped_input_image'], res['template']) +
                                 self.ncc_loss(batch_mri, res['warped_template'])) / 2.
                else:
                    simi_loss = self.ncc_loss(res['warped_input_image'], res['template'])

                simi_loss_item = simi_loss.item()
                total_loss += simi_loss

                if self.args.smooth_reg > 0:
                    if 'disp_i2t' in res:
                        if self.args.smooth_reg_type == 'dvf':
                            smooth_loss = (model.loss.smooth_loss_simple(res['scaled_disp_t2i']) +
                                           model.loss.smooth_loss_simple(res['scaled_disp_i2t'])) / 2.
                        else:
                            smooth_loss = (model.loss.smooth_loss(res['scaled_disp_t2i']) +
                                           model.loss.smooth_loss(res['scaled_disp_i2t'])) / 2.
                    else:
                        if self.args.smooth_reg_type == 'dvf':
                            smooth_loss = model.loss.smooth_loss_simple(res['scaled_disp_t2i'], res['scaled_template'])
                        else:
                            smooth_loss = model.loss.smooth_loss(res['scaled_disp_t2i'],
                                                                 res['scaled_template'])

                    total_loss += self.args.smooth_reg * smooth_loss
                    smooth_loss_item = smooth_loss.item()
                else:
                    smooth_loss_item = 0

                if self.args.cyclic_reg > 0:
                    if 'disp_i2t' in res:
                        cyclic_loss = ((torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5 +
                                       (torch.mean((torch.sum(res['scaled_disp_i2t'], 0)) ** 2)) ** 0.5) / 2.
                    else:
                        cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5
                    total_loss += self.args.cyclic_reg * cyclic_loss
                    cyclic_loss_item = cyclic_loss.item()
                else:
                    cyclic_loss_item = 0

                if 'disp_i2t' in res:
                    disp_i2t = res['disp_i2t']
                else:
                    disp_i2t = self.calcdisp.inverse_disp(res['disp_t2i'])

                composed_disp = self.calcdisp.compose_disp(disp_i2t, res['disp_t2i'], mode='all')
                warped_labels = self.spatial_transform(frame_repeated.to(composed_disp[self.args.ref_frame].dtype),
                                                  composed_disp[:, self.args.ref_frame, ...], mode='nearest').to(batch_label_resampled.dtype)

                mean_reg_dsc_list = self.dice_loss(batch_label, warped_labels, 'reg')
                mean_reg_dsc_loss_item = mean_reg_dsc_list.mean().item()

                if self.args.reg_dsc_weight > 0:
                    total_loss += mean_reg_dsc_list.mean()

                running_total_reg_dsc_loss += mean_reg_dsc_loss_item
                running_reg_dsc_LV += 1.0 - mean_reg_dsc_list[0].item()
                running_reg_dsc_LVM += 1.0 - mean_reg_dsc_list[1].item()
                running_reg_dsc_RV += 1.0 - mean_reg_dsc_list[2].item()
                running_reg_dsc_LA += 1.0 - mean_reg_dsc_list[3].item()
                running_reg_dsc_RA += 1.0 - mean_reg_dsc_list[4].item()

                running_simi_loss += simi_loss_item
                running_smooth_loss += smooth_loss_item
                running_cyclic_loss += cyclic_loss_item

            if self.args.network == 'groupCS':
                if not running_alpha_enc:
                    for j in range(self.args.depth-1):
                        cs_unit_encoder = self.model.unet.cs_unit_encoder[j].clone()
                        cs_unit_decoder = self.model.unet.cs_unit_decoder[j].clone()
                        running_alpha_enc[f'alpha_{j}'] = cs_unit_encoder.cpu().detach().numpy()
                        running_alpha_dec[f'alpha_{j}'] = cs_unit_decoder.cpu().detach().numpy()

                else:
                    for j in range(self.args.depth-1):
                        cs_unit_encoder = self.model.unet.cs_unit_encoder[j].clone()
                        cs_unit_decoder = self.model.unet.cs_unit_decoder[j].clone()
                        running_alpha_enc[f'alpha_{j}'] += cs_unit_encoder.cpu().detach().numpy()
                        running_alpha_dec[f'alpha_{j}'] += cs_unit_decoder.cpu().detach().numpy()

            total_loss_item = total_loss.item()
            running_total_loss += total_loss_item

            # backpropagate and update optimizer learning rate
            total_loss.backward()
            self.optimizer.step()

        epoch_total_loss = running_total_loss / len(self.train_loader)
        self.summary_writer.add_scalars("Losses/total_loss", {'train': epoch_total_loss}, self.current_epoch)

        if 'probs' in res:  # if segmentation task or multi-task
            epoch_total_seg_dsc_loss = running_total_seg_dsc_loss / len(self.train_loader)
            epoch_seg_dsc_LV = running_seg_dsc_LV / len(self.train_loader)
            epoch_seg_dsc_LVM = running_seg_dsc_LVM / len(self.train_loader)
            epoch_seg_dsc_RV = running_seg_dsc_RV / len(self.train_loader)
            epoch_seg_dsc_LA = running_seg_dsc_LA / len(self.train_loader)
            epoch_seg_dsc_RA = running_seg_dsc_RA / len(self.train_loader)
            epoch_focal_loss = running_total_focal_loss / len(self.train_loader)

            self.summary_writer.add_scalars("Losses/total_seg_dsc_loss", {'train': epoch_total_seg_dsc_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/total_focal_loss", {'train': epoch_focal_loss}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/LV", {'train': epoch_seg_dsc_LV}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/LVM", {'train': epoch_seg_dsc_LVM}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/RV", {'train': epoch_seg_dsc_RV}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/LA", {'train': epoch_seg_dsc_LA}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/RA", {'train': epoch_seg_dsc_RA}, self.current_epoch)

            self.logger.info(
                f'Training Seg, {self.current_epoch}, total_loss {epoch_total_loss:.4f}, '
                f'DSC loss {epoch_total_seg_dsc_loss:.4f}, LV_Seg_DSC {epoch_seg_dsc_LV:.3f}, '
                f'LVM_Seg_DSC {epoch_seg_dsc_LVM:.3f}, RV_Seg_DSC {epoch_seg_dsc_RV:.3f}, '
                f'LA_Seg_DSC {epoch_seg_dsc_LA:.3f}, RA_Seg_DSC {epoch_seg_dsc_RA:.3f}')

        if 'disp_t2i' in res:  # if registration task or multi-task

            epoch_simi_loss = running_simi_loss / len(self.train_loader)
            epoch_smooth_loss = running_smooth_loss / len(self.train_loader)
            epoch_cyclic_loss = running_cyclic_loss / len(self.train_loader)
            epoch_total_reg_dsc_loss = running_total_reg_dsc_loss / len(self.train_loader)
            epoch_reg_dsc_LV = running_reg_dsc_LV / len(self.train_loader)
            epoch_reg_dsc_LVM = running_reg_dsc_LVM / len(self.train_loader)
            epoch_reg_dsc_RV = running_reg_dsc_RV / len(self.train_loader)
            epoch_reg_dsc_LA = running_reg_dsc_LA / len(self.train_loader)
            epoch_reg_dsc_RA = running_reg_dsc_RA / len(self.train_loader)

            self.summary_writer.add_scalars("Losses/similarity_loss", {'train': epoch_simi_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/smooth_loss", {'train': epoch_smooth_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/cyclic_loss", {'train': epoch_cyclic_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/total_reg_dsc_loss", {'train': epoch_total_reg_dsc_loss}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/LV", {'train': epoch_reg_dsc_LV}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/LVM", {'train': epoch_reg_dsc_LVM}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/RV", {'train': epoch_reg_dsc_RV}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/LA", {'train': epoch_reg_dsc_LA}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/RA", {'train': epoch_reg_dsc_RA}, self.current_epoch)


            self.logger.info( f'Training Reg, {self.current_epoch}, total loss {epoch_total_loss:.4f}, '
                              f'simi. loss {epoch_simi_loss:.4f}, DSC loss {epoch_total_reg_dsc_loss:.4f}, '
                              f'smooth loss {epoch_smooth_loss:.4f}, cyclic loss {epoch_cyclic_loss:.4f}, '
                              f'LV_Reg_DSC {epoch_reg_dsc_LV:.3f}, LVM_Reg_DSC {epoch_reg_dsc_LVM:.3f}, '
                              f'RV_Reg_DSC {epoch_reg_dsc_RV:.3f}, LA_Reg_DSC {epoch_reg_dsc_LA:.3f}, '
                              f'RA_Reg_DSC {epoch_reg_dsc_RA:.3f}')

            if self.args.network == 'groupCS':
                epoch_alpha_enc = {}
                epoch_alpha_dec = {}
                for j in range(self.args.depth - 1):
                    epoch_alpha_enc[f'alpha_{j}'] = np.mean(running_alpha_enc[f'alpha_{j}']/len(self.train_loader), axis=0)
                    epoch_alpha_dec[f'alpha_{j}'] = np.mean(running_alpha_dec[f'alpha_{j}']/len(self.train_loader), axis=0)

                    self.logger.info(f"Encoder #{j}: alpha00 {epoch_alpha_enc[f'alpha_{j}'][0, 0]:.4f}, "
                                     f"alpha01 {epoch_alpha_enc[f'alpha_{j}'][0, 1]:.4f}, "
                                     f"alpha10 {epoch_alpha_enc[f'alpha_{j}'][1, 0]:.4f}, "
                                     f"alpha11 {epoch_alpha_enc[f'alpha_{j}'][1, 1]:.4f}, ")

                    self.logger.info(f"Decoder #{j}: alpha00 {epoch_alpha_dec[f'alpha_{j}'][0, 0]:.4f}, "
                                     f"alpha01 {epoch_alpha_dec[f'alpha_{j}'][0, 1]:.4f}, "
                                     f"alpha10 {epoch_alpha_dec[f'alpha_{j}'][1, 0]:.4f}, "
                                     f"alpha11 {epoch_alpha_dec[f'alpha_{j}'][1, 1]:.4f}, ")

                    self.summary_writer.add_scalars(f"CS/alpha_enc_{j}_00", {'train': epoch_alpha_enc[f'alpha_{j}'][0, 0]}, self.current_epoch)
                    self.summary_writer.add_scalars(f"CS/alpha_enc_{j}_01", {'train': epoch_alpha_enc[f'alpha_{j}'][0, 1]}, self.current_epoch)
                    self.summary_writer.add_scalars(f"CS/alpha_enc_{j}_10", {'train': epoch_alpha_enc[f'alpha_{j}'][1, 0]}, self.current_epoch)
                    self.summary_writer.add_scalars(f"CS/alpha_enc_{j}_11", {'train': epoch_alpha_enc[f'alpha_{j}'][1, 1]}, self.current_epoch)

                    self.summary_writer.add_scalars(f"CS/alpha_dec_{j}_00", {'train': epoch_alpha_dec[f'alpha_{j}'][0, 0]}, self.current_epoch)
                    self.summary_writer.add_scalars(f"CS/alpha_dec_{j}_01", {'train': epoch_alpha_dec[f'alpha_{j}'][0, 1]}, self.current_epoch)
                    self.summary_writer.add_scalars(f"CS/alpha_dec_{j}_10", {'train': epoch_alpha_dec[f'alpha_{j}'][1, 0]}, self.current_epoch)
                    self.summary_writer.add_scalars(f"CS/alpha_dec_{j}_11", {'train': epoch_alpha_dec[f'alpha_{j}'][1, 1]}, self.current_epoch)




    def validate(self):

        # Set model to evaluation mode
        self.model.eval()
        # initialize stats
        running_total_loss = 0.
        running_simi_loss = 0.
        running_cyclic_loss = 0.
        running_smooth_loss = 0.
        running_total_seg_dsc_loss = 0.
        running_total_reg_dsc_loss = 0.
        running_total_focal_loss = 0.

        running_seg_dsc_LV = 0.
        running_seg_dsc_LVM = 0.
        running_seg_dsc_RV = 0.
        running_seg_dsc_LA = 0.
        running_seg_dsc_RA = 0.

        running_reg_dsc_LV = 0.
        running_reg_dsc_LVM = 0.
        running_reg_dsc_RV = 0.
        running_reg_dsc_LA = 0.
        running_reg_dsc_RA = 0.
        i = 1

        with torch.no_grad():
            for val_batch in self.validation_loader:
                total_loss = 0.

                batch_mri_init = val_batch['mri'][tio.DATA].to(device)[0, ...]
                batch_label_init = val_batch['heart'][tio.DATA].to(device)[0, ...]

                batch_mri_resampled = F.interpolate(batch_mri_init.unsqueeze(0), (self.args.image_shape[0],
                                                                                  self.args.image_shape[1],
                                                                                  self.args.num_images_per_group),
                                                    mode='trilinear', align_corners=True).squeeze(0)

                batch_label_resampled = F.interpolate(batch_label_init.unsqueeze(0), (self.args.image_shape[0],
                                                                                      self.args.image_shape[1],
                                                                                      self.args.num_images_per_group),
                                                      mode='nearest').squeeze(0)

                batch_mri = batch_mri_resampled.permute(3, 0, 2, 1)  # (n,1,h,w)
                batch_label = batch_label_resampled.permute(3, 0, 2, 1)  # (n,1,h,w)
                copy_batch_label = batch_label.clone()[self.args.ref_frame, ...].unsqueeze(0)
                frame_repeated = copy_batch_label.repeat(self.args.num_images_per_group, 1, 1, 1)

                res = self.model(batch_mri)

                if 'probs' in res:  # if segmentation task or multi-task

                    mean_seg_dsc_list = self.dice_loss(batch_label, res['probs'], 'seg')
                    mean_seg_dsc_loss_item = mean_seg_dsc_list.mean().item()
                    total_loss += mean_seg_dsc_list.mean()

                    if self.args.focal_weight > 0:
                        mean_focal_list = self.focal_loss(batch_label, res['probs'])
                        mean_focal_loss_item = mean_focal_list.mean().item()
                        total_loss += self.args.focal_weight * mean_focal_list.mean()
                    else:
                        mean_focal_loss_item = 0

                    running_total_focal_loss += mean_focal_loss_item
                    running_total_seg_dsc_loss += mean_seg_dsc_loss_item
                    running_seg_dsc_LV += 1.0 - mean_seg_dsc_list[0].item()
                    running_seg_dsc_LVM += 1.0 - mean_seg_dsc_list[1].item()
                    running_seg_dsc_RV += 1.0 - mean_seg_dsc_list[2].item()
                    running_seg_dsc_LA += 1.0 - mean_seg_dsc_list[3].item()
                    running_seg_dsc_RA += 1.0 - mean_seg_dsc_list[4].item()

                if 'disp_t2i' in res:  # if registration task or multi-task

                    if 'disp_i2t' in res:
                        simi_loss = (self.ncc_loss(res['warped_input_image'], res['template']) +
                                     self.ncc_loss(batch_mri, res['warped_template'])) / 2.
                    else:
                        simi_loss = self.ncc_loss(res['warped_input_image'], res['template'])

                    simi_loss_item = simi_loss.item()
                    total_loss += simi_loss

                    if self.args.smooth_reg > 0:
                        if 'disp_i2t' in res:
                            if self.args.smooth_reg_type == 'dvf':
                                smooth_loss = (model.loss.smooth_loss_simple(res['scaled_disp_t2i']) +
                                               model.loss.smooth_loss_simple(res['scaled_disp_i2t'])) / 2.
                            else:
                                smooth_loss = (model.loss.smooth_loss(res['scaled_disp_t2i']) +
                                               model.loss.smooth_loss(res['scaled_disp_i2t'])) / 2.
                        else:
                            if self.args.smooth_reg_type == 'dvf':
                                smooth_loss = model.loss.smooth_loss_simple(res['scaled_disp_t2i'], res['scaled_template'])
                            else:
                                smooth_loss = model.loss.smooth_loss(res['scaled_disp_t2i'],
                                                                            res['scaled_template'])

                        total_loss += self.args.smooth_reg * smooth_loss
                        smooth_loss_item = smooth_loss.item()
                    else:
                        smooth_loss_item = 0

                    if self.args.cyclic_reg > 0:
                        if 'disp_i2t' in res:
                            cyclic_loss = ((torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5 +
                                           (torch.mean((torch.sum(res['scaled_disp_i2t'], 0)) ** 2)) ** 0.5) / 2.
                        else:
                            cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5
                        total_loss += self.args.cyclic_reg * cyclic_loss
                        cyclic_loss_item = cyclic_loss.item()
                    else:
                        cyclic_loss_item = 0

                    if 'disp_i2t' in res:
                        disp_i2t = res['disp_i2t']
                    else:
                        disp_i2t = self.calcdisp.inverse_disp(res['disp_t2i'])
                    composed_disp = self.calcdisp.compose_disp(disp_i2t, res['disp_t2i'], mode='all')
                    warped_labels = self.spatial_transform(frame_repeated.to(composed_disp[self.args.ref_frame].dtype),
                                                           composed_disp[:, self.args.ref_frame, ...], mode='nearest').to(
                        batch_label_resampled.dtype)

                    mean_reg_dsc_list = self.dice_loss(batch_label, warped_labels, 'reg')
                    mean_reg_dsc_loss_item = mean_reg_dsc_list.mean().item()

                    if self.args.reg_dsc_weight > 0:
                        total_loss += mean_reg_dsc_list.mean()

                    running_total_reg_dsc_loss += mean_reg_dsc_loss_item
                    running_reg_dsc_LV += 1.0 - mean_reg_dsc_list[0].item()
                    running_reg_dsc_LVM += 1.0 - mean_reg_dsc_list[1].item()
                    running_reg_dsc_RV += 1.0 - mean_reg_dsc_list[2].item()
                    running_reg_dsc_LA += 1.0 - mean_reg_dsc_list[3].item()
                    running_reg_dsc_RA += 1.0 - mean_reg_dsc_list[4].item()

                    running_simi_loss += simi_loss_item
                    running_smooth_loss += smooth_loss_item
                    running_cyclic_loss += cyclic_loss_item

                total_loss_item = total_loss.item()
                running_total_loss += total_loss_item

                if self.current_epoch % self.args.save_image_every == 0:

                    grid_input_img = torchvision.utils.make_grid(batch_mri, nrow=5)
                    grid_input_label = torchvision.utils.make_grid(batch_label.to(torch.float), nrow=5, normalize=True)
                    self.summary_writer.add_image(f'inputImage_{i}', grid_input_img, self.current_epoch)
                    self.summary_writer.add_image(f'inputLabel_{i}', grid_input_label, self.current_epoch)


                    if 'probs' in res:  # if segmentation task or multi-task
                        grid_predicted_label = torchvision.utils.make_grid(res['predicted_label'].to(torch.float),
                                                                           nrow=5, normalize=True)
                        self.summary_writer.add_image(f'predictedLabel_{i}', grid_predicted_label,
                                                      self.current_epoch)

                    if 'disp_t2i' in res:
                        grid_warped_img = torchvision.utils.make_grid(res['warped_input_image'], nrow=5)
                        grid_warped_label = torchvision.utils.make_grid(warped_labels.to(torch.float), nrow=5,
                                                                        normalize=True)

                        grid_temp = torchvision.utils.make_grid(res['template'], nrow=1)
                        self.summary_writer.add_image(f'warpedImage_{i}', grid_warped_img)
                        self.summary_writer.add_image(f'warpedLabel_{i}', grid_warped_label, self.current_epoch)
                        self.summary_writer.add_image(f'templateImage_{i}', grid_temp, self.current_epoch)
                        # self.summary_writer.add_figure(f'DVF_{i}', flow_color(res['disp_t2i'].clone()), self.current_epoch)

                    # plt.imshow(grid_img.cpu().permute(1, 2, 0))
                    # plt.axis('off')
                    i += 1

        epoch_total_loss = running_total_loss / len(self.validation_loader)
        self.summary_writer.add_scalars("Losses/total_loss", {'validation': epoch_total_loss}, self.current_epoch)

        if 'probs' in res:  # if segmentation task or multi-task
            epoch_total_seg_dsc_loss = running_total_seg_dsc_loss / len(self.validation_loader)
            epoch_seg_dsc_LV = running_seg_dsc_LV / len(self.validation_loader)
            epoch_seg_dsc_LVM = running_seg_dsc_LVM / len(self.validation_loader)
            epoch_seg_dsc_RV = running_seg_dsc_RV / len(self.validation_loader)
            epoch_seg_dsc_LA = running_seg_dsc_LA / len(self.validation_loader)
            epoch_seg_dsc_RA = running_seg_dsc_RA / len(self.validation_loader)
            epoch_focal_loss = running_total_focal_loss / len(self.validation_loader)

            self.summary_writer.add_scalars("Losses/total_seg_dsc_loss", {'validation': epoch_total_seg_dsc_loss},
                                            self.current_epoch)
            self.summary_writer.add_scalars("Losses/total_focal_loss", {'validation': epoch_focal_loss}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/LV", {'validation': epoch_seg_dsc_LV}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/LVM", {'validation': epoch_seg_dsc_LVM}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/RV", {'validation': epoch_seg_dsc_RV}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/LA", {'validation': epoch_seg_dsc_LA}, self.current_epoch)
            self.summary_writer.add_scalars("SegDice/RA", {'validation': epoch_seg_dsc_RA}, self.current_epoch)

            self.logger.info(
                f'Validation Seg, {self.current_epoch}, total_loss {epoch_total_loss:.4f}, LV_Seg_DSC {epoch_seg_dsc_LV:.3f}, '
                f'LVM_Seg_DSC {epoch_seg_dsc_LVM:.3f}, RV_Seg_DSC {epoch_seg_dsc_RV:.3f}, LA_Seg_DSC {epoch_seg_dsc_LA:.3f}, '
                f'RA_Seg_DSC {epoch_seg_dsc_RA:.3f}')

        if 'disp_t2i' in res:  # if registration task or multi-task

            epoch_simi_loss = running_simi_loss / len(self.validation_loader)
            epoch_smooth_loss = running_smooth_loss / len(self.validation_loader)
            epoch_cyclic_loss = running_cyclic_loss / len(self.validation_loader)
            epoch_total_reg_dsc_loss = running_total_reg_dsc_loss / len(self.validation_loader)
            epoch_reg_dsc_LV = running_reg_dsc_LV / len(self.validation_loader)
            epoch_reg_dsc_LVM = running_reg_dsc_LVM / len(self.validation_loader)
            epoch_reg_dsc_RV = running_reg_dsc_RV / len(self.validation_loader)
            epoch_reg_dsc_LA = running_reg_dsc_LA / len(self.validation_loader)
            epoch_reg_dsc_RA = running_reg_dsc_RA / len(self.validation_loader)

            self.summary_writer.add_scalars("Losses/similarity_loss", {'validation': epoch_simi_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/smooth_loss", {'validation': epoch_smooth_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/cyclic_loss", {'validation': epoch_cyclic_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/total_reg_dsc_loss", {'validation': epoch_total_reg_dsc_loss},
                                            self.current_epoch)
            self.summary_writer.add_scalars("RegDice/LV", {'validation': epoch_reg_dsc_LV}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/LVM", {'validation': epoch_reg_dsc_LVM}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/RV", {'validation': epoch_reg_dsc_RV}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/LA", {'validation': epoch_reg_dsc_LA}, self.current_epoch)
            self.summary_writer.add_scalars("RegDice/RA", {'validation': epoch_reg_dsc_RA}, self.current_epoch)


            self.logger.info(
                f'Validation Reg, {self.current_epoch}, total loss {epoch_total_loss:.4f}, '
                f'simi. loss {epoch_simi_loss:.4f}, smooth loss {epoch_smooth_loss:.4f}, '
                f'DSC loss {epoch_total_reg_dsc_loss:.4f}, cyclic loss {epoch_cyclic_loss:.3f}, '
                f'LV_Reg_DSC {epoch_reg_dsc_LV:.3f}, LVM_Reg_DSC {epoch_reg_dsc_LVM:.3f}, '
                f'RV_Reg_DSC {epoch_reg_dsc_RV:.3f}, LA_Reg_DSC {epoch_reg_dsc_LA:.3f}, '
                f'RA_Reg_DSC {epoch_reg_dsc_RA:.3f}')


    def inference(self):

        if self.args.inference_set == 'validation':
            loader = self.validation_loader
        elif self.args.inference_set == 'test':
            loader = self.test_loader

        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for test_batch in loader:

                batch_mri_init = test_batch['mri'][tio.DATA].to(device)[0, ...]
                batch_label_init = test_batch['heart'][tio.DATA].to(device)[0, ...]

                batch_mri_resampled = F.interpolate(batch_mri_init.unsqueeze(0), (self.args.image_shape[0],
                                                                                  self.args.image_shape[1],
                                                                                  self.args.num_images_per_group),
                                                    mode='trilinear', align_corners=True).squeeze(0)

                batch_label_resampled = F.interpolate(batch_label_init.unsqueeze(0), (self.args.image_shape[0],
                                                                                      self.args.image_shape[1],
                                                                                      self.args.num_images_per_group),
                                                      mode='nearest').squeeze(0)

                batch_mri = batch_mri_resampled.permute(3, 0, 2, 1)  # (n,1,h,w)
                batch_label = batch_label_resampled.permute(3, 0, 2, 1)  # (n,1,h,w)
                copy_batch_label = batch_label.clone()[self.args.ref_frame, ...].unsqueeze(0)
                frame_repeated = copy_batch_label.repeat(self.args.num_images_per_group, 1, 1, 1)

                res = self.model(batch_mri)

                patient_name = test_batch["mri"][tio.PATH][0].split('/')[-1].split('.mha')[0]
                patient_output_path = os.path.join(self.args.output_dir, self.args.inference_set, patient_name)
                if not os.path.exists(patient_output_path):
                    os.makedirs(patient_output_path)

                if 'probs' in res:
                    batch_predicted_label_resampled = F.interpolate(res['predicted_label'].permute(1, 2, 3, 0).
                                                                    unsqueeze(0).to(torch.float),
                                                                    (self.args.image_shape[0], self.args.image_shape[1],
                                                                     batch_label_init.shape[-1]), mode='nearest').squeeze(0)

                    sitk_output_label = sitk.GetImageFromArray(batch_predicted_label_resampled.squeeze().
                                                               permute(2, 0, 1).cpu())
                    sitk_output_label.CopyInformation(sitk.ReadImage(test_batch["mri"][tio.PATH][0]))
                    sitk.WriteImage(sitk_output_label, os.path.join(patient_output_path, 'predicted_label.mha'))

                if 'disp_t2i' in res:
                    if 'disp_i2t' in res:
                        disp_i2t = res['disp_i2t']
                    else:
                        disp_i2t = self.calcdisp.inverse_disp(res['disp_t2i'])
                    composed_disp = self.calcdisp.compose_disp(disp_i2t, res['disp_t2i'], mode='all')
                    warped_labels = self.spatial_transform(frame_repeated.to(composed_disp[self.args.ref_frame].dtype),
                                                           composed_disp[:, self.args.ref_frame, ...],
                                                           mode='nearest').to(batch_label_resampled.dtype)

                    copy_warped_label = warped_labels.clone().detach()
                    copy_disp_t2i = res['disp_t2i'].clone().detach()

                    batch_warped_label_resampled = F.interpolate(copy_warped_label.permute(1, 2, 3, 0).unsqueeze(0),
                                                                 (self.args.image_shape[0], self.args.image_shape[1],
                                                                  batch_label_init.shape[-1]),
                                                                 mode='nearest').squeeze(0)

                    batch_disp_t2i_resampled1 = F.interpolate(copy_disp_t2i[:, 0, ...].unsqueeze(1).permute(1, 2, 3, 0).unsqueeze(0),
                                                             (self.args.image_shape[0], self.args.image_shape[1],
                                                                  batch_label_init.shape[-1]),
                                                             mode='trilinear').squeeze(0)

                    batch_disp_t2i_resampled2 = F.interpolate(copy_disp_t2i[:, 1, ...].unsqueeze(1).permute(1, 2, 3, 0).unsqueeze(0),
                                                             (self.args.image_shape[0], self.args.image_shape[1],
                                                                  batch_label_init.shape[-1]),
                                                             mode='trilinear').squeeze(0)

                    batch_disp_t2i_resampled = torch.cat((batch_disp_t2i_resampled1, batch_disp_t2i_resampled2), dim=0)

                    sitk_output_label = sitk.GetImageFromArray(batch_warped_label_resampled.squeeze().permute(2, 0, 1).cpu())
                    sitk_output_label.CopyInformation(sitk.ReadImage(test_batch["mri"][tio.PATH][0]))
                    sitk.WriteImage(sitk_output_label, os.path.join(patient_output_path, 'warped_label.mha'))

                    sitk_output_dvf = sitk.GetImageFromArray(batch_disp_t2i_resampled.permute(3, 1, 2, 0).cpu(), isVector=True)
                    sitk_output_dvf.SetSpacing(sitk.ReadImage(test_batch["mri"][tio.PATH][0]).GetSpacing())
                    sitk_output_dvf.SetDirection(sitk.ReadImage(test_batch["mri"][tio.PATH][0]).GetDirection())
                    sitk.WriteImage(sitk_output_dvf, os.path.join(patient_output_path, 'dvf.mha'))


                print(f'finished patient {patient_name}')


    def eval(self):

        patientNameList_tmp = sorted(os.listdir(os.path.join(self.args.output_dir, self.args.eval_set)))
        patientNameList = [x for x in patientNameList_tmp if os.path.isdir(os.path.join(self.args.output_dir,
                                                                                        self.args.eval_set, x))]
        for i in range(len(self.args.label_name)):

            patient_list = []
            data_dict = {}
            data_dict['LV'] = {'idx': 1, 'dsc': np.ones((len(patientNameList), 30)) * -1,
                               'cd': np.ones((len(patientNameList), 30)) * -1,
                               'hd': np.ones((len(patientNameList), 30)) * -1}

            data_dict['LVM'] = {'idx': 2, 'dsc': np.ones((len(patientNameList), 30)) * -1,
                                'cd': np.ones((len(patientNameList), 30)) * -1,
                                'hd': np.ones((len(patientNameList), 30)) * -1}

            data_dict['RV'] = {'idx': 3, 'dsc': np.ones((len(patientNameList), 30)) * -1,
                               'cd': np.ones((len(patientNameList), 30)) * -1,
                               'hd': np.ones((len(patientNameList), 30)) * -1}

            data_dict['LA'] = {'idx': 4, 'dsc': np.ones((len(patientNameList), 30)) * -1,
                               'cd': np.ones((len(patientNameList), 30)) * -1,
                               'hd': np.ones((len(patientNameList), 30)) * -1}

            data_dict['RA'] = {'idx': 5, 'dsc': np.ones((len(patientNameList), 30)) * -1,
                               'cd': np.ones((len(patientNameList), 30)) * -1,
                               'hd': np.ones((len(patientNameList), 30)) * -1}

            xlsx_writer = pd.ExcelWriter(os.path.join(self.args.output_dir, self.args.eval_set,
                                                      f'{self.args.excel_tab_name[i]}.xlsx'), engine='xlsxwriter')  # excel sheet name

            for j in tqdm(range(len(patientNameList)), desc=f'Evaluation Progress'):


                if self.args.debug:
                    gt_mask_path = os.path.join(self.args.validation_data_folder, 'masks', patientNameList[j] + '.mha')
                elif self.args.eval_set == 'test':
                    gt_mask_path = os.path.join(self.args.test_data_folder, 'masks', patientNameList[j] + '.mha')
                elif self.args.eval_set == 'validation':
                    gt_mask_path = os.path.join(self.args.validation_data_folder, 'masks', patientNameList[j] + '.mha')

                predicted_mask_path = os.path.join(self.args.output_dir, self.args.eval_set, patientNameList[j],
                                                   self.args.label_name[i])

                groundtruth_mask = sitk.GetArrayFromImage(sitk.ReadImage(gt_mask_path)).transpose(1, 2, 0)
                estimated_mask = sitk.GetArrayFromImage(sitk.ReadImage(predicted_mask_path)).transpose(1, 2, 0)

                for label in list(data_dict.keys()):
                    dsc = categorical_dice_stack(groundtruth_mask, estimated_mask, label_class=data_dict[label]['idx'])
                    cd, hd= contour_distances_stack(groundtruth_mask, estimated_mask, label_class=data_dict[label]['idx'])

                    data_dict[label]['dsc'][j, :groundtruth_mask.shape[-1]] = dsc
                    data_dict[label]['cd'][j, :groundtruth_mask.shape[-1]] = cd
                    data_dict[label]['hd'][j, :groundtruth_mask.shape[-1]] = hd


                patient_list.append(patientNameList[j])

            for label in list(data_dict.keys()):
                for metric in list(data_dict[label].keys())[1:]:
                    header = [f'frame_{x}' for x in range(data_dict[label][metric].shape[1])]
                    df = pd.DataFrame(data_dict[label][metric], columns=header)
                    df = df.replace(-1, np.NaN)
                    df['Mean'] = df.mean(axis=1, skipna=True)
                    df['Std'] = df.std(axis=1, skipna=True)
                    df['Patient'] = patient_list
                    df = df.reindex(['Patient', 'Mean', 'Std'] + header, axis=1)
                    df.to_excel(xlsx_writer, sheet_name=f'{label}_{metric}')
            xlsx_writer.save()




    def eval_old(self):

        xlsx_writer = pd.ExcelWriter(os.path.join(self.args.output_dir, f'Network_Evaluation.xlsx'),
                                engine='xlsxwriter')  # excel sheet name

        patientNameList = sorted(os.listdir(os.path.join(self.args.output_dir)))

        for i in range(len(self.args.label_name)):

            patient_list = []
            mean_mcd_list_LV = []
            mean_mcd_list_LVM = []
            mean_mcd_list_RV = []
            mean_mcd_list_LA = []
            mean_mcd_list_RA = []

            mean_hd_list_LV = []
            mean_hd_list_LVM = []
            mean_hd_list_RV = []
            mean_hd_list_LA = []
            mean_hd_list_RA = []

            mean_dsc_list_LV = []
            mean_dsc_list_LVM = []
            mean_dsc_list_RV = []
            mean_dsc_list_LA = []
            mean_dsc_list_RA = []



            for j in tqdm(range(len(patientNameList)), desc=f'Evaluation Progress'):


                if self.args.debug:
                    gt_mask_path = os.path.join(self.args.validation_data_folder, 'masks', patientNameList[j] + '.mha')
                else:
                    gt_mask_path = os.path.join(self.args.test_data_folder, 'masks', patientNameList[j] + '.mha')

                predicted_mask_path = os.path.join(self.args.output_dir, patientNameList[j], self.args.label_name[i])

                groundtruth_mask = sitk.GetArrayFromImage(sitk.ReadImage(gt_mask_path)).transpose(1, 2, 0)
                estimated_mask = sitk.GetArrayFromImage(sitk.ReadImage(predicted_mask_path)).transpose(1, 2, 0)


                mean_dsc_LV = categorical_dice_stack(groundtruth_mask, estimated_mask, label_class=1)
                mean_dsc_LVM = categorical_dice_stack(groundtruth_mask, estimated_mask, label_class=2)
                mean_dsc_RV = categorical_dice_stack(groundtruth_mask, estimated_mask, label_class=3)
                mean_dsc_LA = categorical_dice_stack(groundtruth_mask, estimated_mask, label_class=4)
                mean_dsc_RA = categorical_dice_stack(groundtruth_mask, estimated_mask, label_class=5)

                mean_mcd_LV, mean_hd_LV = contour_distances_stack(groundtruth_mask, estimated_mask, label_class=1)
                mean_mcd_LVM, mean_hd_LVM = contour_distances_stack(groundtruth_mask, estimated_mask, label_class=2)
                mean_mcd_RV, mean_hd_RV = contour_distances_stack(groundtruth_mask, estimated_mask, label_class=3)
                mean_mcd_LA, mean_hd_LA = contour_distances_stack(groundtruth_mask, estimated_mask, label_class=4)
                mean_mcd_RA, mean_hd_RA = contour_distances_stack(groundtruth_mask, estimated_mask, label_class=5)

                mean_mcd_list_LV.append(mean_mcd_LV)
                mean_mcd_list_LVM.append(mean_mcd_LVM)
                mean_mcd_list_RV.append(mean_mcd_RV)
                mean_mcd_list_LA.append(mean_mcd_LA)
                mean_mcd_list_RA.append(mean_mcd_RA)

                mean_hd_list_LV.append(mean_hd_LV)
                mean_hd_list_LVM.append(mean_hd_LVM)
                mean_hd_list_RV.append(mean_hd_RV)
                mean_hd_list_LA.append(mean_hd_LA)
                mean_hd_list_RA.append(mean_hd_RA)

                mean_dsc_list_LV.append(mean_dsc_LV)
                mean_dsc_list_LVM.append(mean_dsc_LVM)
                mean_dsc_list_RV.append(mean_dsc_RV)
                mean_dsc_list_LA.append(mean_dsc_LA)
                mean_dsc_list_RA.append(mean_dsc_RA)

                patient_list.append(patientNameList[j])


            data = {'Patient': patient_list,
                    'LV_MCD': mean_mcd_list_LV, 'LV_HD': mean_hd_list_LV,
                    'LV_DSC': mean_dsc_list_LV,
                    'LVM_MCD': mean_mcd_list_LVM, 'LVM_HD': mean_hd_list_LVM,
                    'LVM_DSC': mean_dsc_list_LVM,
                    'RV_MCD': mean_mcd_list_RV, 'RV_HD': mean_hd_list_RV,
                    'RV_DSC': mean_dsc_list_RV,
                    'LA_MCD': mean_mcd_list_LA, 'LA_HD': mean_hd_list_LA,
                    'LA_DSC': mean_dsc_list_LA,
                    'RA_MCD': mean_mcd_list_RA, 'RA_HD': mean_hd_list_RA,
                    'RA_DSC': mean_dsc_list_RA}

            df = pd.DataFrame(data, dtype=float)
            df = df.reindex(['Patient', 'LV_DSC', 'LV_MCD', 'LV_HD',
                                  'LVM_DSC', 'LVM_MCD', 'LVM_HD',
                                  'RV_DSC', 'RV_MCD', 'RV_HD',
                                  'LA_DSC', 'LA_MCD', 'LA_HD',
                                  'RA_DSC', 'RA_MCD', 'RA_HD'], axis=1)
            df.to_excel(xlsx_writer, sheet_name=self.args.excel_tab_name[i])

        xlsx_writer.save()

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        if self.args.debug or self.args.mode != 'train':
            pass
        else:
            self.logger.info("Please wait while finalizing the operation.. Thank you")
            self.summary_writer.export_scalars_to_json(os.path.join(self.args.tensorboard_dir, "all_scalars.json"))
            self.summary_writer.close()
