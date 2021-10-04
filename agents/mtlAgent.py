import logging
import os
import time

import SimpleITK as sitk
# from utils.utils.contour_eval import *
import torchio as tio
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from agents.base import BaseAgent
from graphs.losses.loss import *
from graphs.models.regnet import RegNet
from graphs.models.segnet import SegNet
from utils import dataset_niftynet as dset_utils
from utils.SpatialTransformer import SpatialTransformer
from utils.model_util import count_parameters
from utils.segmentation_eval import evaluation
from utils.util import clean_data


class mtlAgent(BaseAgent):
    def __init__(self, args, data_config):
        super(mtlAgent).__init__()

        self.args = args
        self.data_config = data_config
        self.logger = logging.getLogger()
        self.current_epoch = 0
        self.current_iteration = 0

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
                self.model = SegNet(in_channels=len(self.args.input), classes=self.args.num_classes,
                                    depth=self.args.depth, initial_channels=self.args.initial_channels,
                                    channels_list = self.args.num_featurmaps).to(self.args.device)
                # Create instance from the loss
                self.dsc_loss = Multi_DSC_Loss().to(self.args.device)
            elif self.args.network == 'Reg':
                self.model = RegNet(in_channels=len(self.args.input), dim=self.args.num_classes,
                                    depth=self.args.depth, initial_channels=self.args.initial_channels,
                                    channels_list=self.args.num_featurmaps).to(self.args.device)
                # Create instance from the loss
                self.ncc_loss = NCC(self.args.dim, self.args.ncc_window_size).to(self.args.device)
                self.smooth_loss = GradientSmoothing(energy_type='bending')
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
            checkpoint = torch.load(filename, map_location=self.args.device)

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

        # initialize stats
        running_loss = 0.
        running_ncc_loss = 0.
        running_dvf_loss = 0.
        running_seg_dsc_loss = 0.
        running_reg_dsc_loss = 0.
        running_seg_dsc = 0.
        running_reg_dsc = 0.
        running_ncc = 0.
        epoch_samples = 0

        for batch_idx, (fimage, flabel, mimage, mlabel) in enumerate(self.dataloaders['training'], 1):
            # switch model to training mode, clear gradient accumulators
            self.model.train()
            self.optimizer.zero_grad()
            self.model.zero_grad()

            data_dict = clean_data(fimage, flabel, mimage, mlabel, self.args)
            nbatches, wsize, nchannels, x, y, z, _ = fimage.size()

            # forward pass
            if self.args.network == 'Seg':
                if self.args.in_channels_seg == 1:
                    res = self.model(data_dict['fimage'])
                elif self.args.in_channels_seg == 2 and 'Im' in self.args.input_segmentation:
                    res = self.model(data_dict['fimage'], data_dict['mimage'])
                elif self.args.in_channels_seg == 2 and 'Sm' in self.args.input_segmentation:
                    res = self.model(data_dict['fimage'], data_dict['mlabel'])
                elif self.args.in_channels_seg == 3:
                    res = self.model(data_dict['fimage'], data_dict['mimage'], data_dict['mlabel'])
                else:
                    self.logger.error(self.args.input_segmentation, "wrong input")

                seg_dsc_loss_high, seg_dsc_list_high = self.dsc_loss(data_dict['flabel_high'], res['x_high_res'])
                seg_dsc_loss_mid, seg_dsc_list_mid = self.dsc_loss(data_dict['flabel_mid'], res['x_mid_res'])
                seg_dsc_loss_low, seg_dsc_list_low = self.dsc_loss(data_dict['flabel_low'], res['x_low_res'])

                seg_dsc_loss = self.args.level_weights[0] * seg_dsc_loss_high + \
                               self.args.level_weights[1] * seg_dsc_loss_mid + \
                               self.args.level_weights[2] * seg_dsc_loss_low

                loss = seg_dsc_loss


            elif self.args.network == 'Reg':
                if self.args.in_channels_reg == 2:
                    res = self.model(data_dict['fimage'], data_dict['mimage'])
                elif self.args.in_channels_reg == 3:
                    res = self.model(data_dict['fimage'], data_dict['mimage'], data_dict['mlabel'])

                mimage_high_out = self.spatial_transform(data_dict['mimage_high'], res['high_res_dvf'])
                mimage_mid_out = self.spatial_transform(data_dict['mimage_mid'], res['mid_res_dvf'])
                mimage_low_out = self.spatial_transform(data_dict['mimage_low'], res['low_res_dvf'])


                mlabel_high_out = self.spatial_transform(data_dict['mlabel_high_hot'], res['high_res_dvf'], mode='nearest')
                mlabel_mid_out = self.spatial_transform(data_dict['mlabel_mid_hot'], res['mid_res_dvf'], mode='nearest')
                mlabel_low_out = self.spatial_transform(data_dict['mlabel_low_hot'], res['low_res_dvf'], mode='nearest')

                reg_dsc_loss_high, reg_dsc_list_high = self.dsc_loss(data_dict['flabel_high'], mlabel_high_out,
                                                                     use_activation=False)
                reg_dsc_loss_mid, reg_dsc_list_mid = self.dsc_loss(data_dict['flabel_mid'], mlabel_mid_out,
                                                                   use_activation=False)
                reg_dsc_loss_low, reg_dsc_list_low = self.dsc_loss(data_dict['flabel_low'], mlabel_low_out,
                                                                   use_activation=False)


                ncc_loss_high = self.ncc_loss(data_dict['fimage_high'], mimage_high_out)
                ncc_loss_mid = self.ncc_loss(data_dict['fimage_mid'], mimage_mid_out)
                ncc_loss_low = self.ncc_loss(data_dict['fimage_low'], mimage_low_out)

                dvf_loss_high = self.smooth_loss(res['high_res_dvf'])
                dvf_loss_mid = self.smooth_loss(res['mid_res_dvf'])
                dvf_loss_low = self.smooth_loss(res['low_res_dvf'])

                reg_dsc_loss = self.args.level_weights[0] * reg_dsc_loss_high + \
                               self.args.level_weights[1] * reg_dsc_loss_mid + \
                               self.args.level_weights[2] * reg_dsc_loss_low

                ncc_loss = self.args.level_weights[0] * ncc_loss_high + \
                           self.args.level_weights[1] * ncc_loss_mid + \
                           self.args.level_weights[2] * ncc_loss_low

                dvf_loss = self.args.level_weights[0] * dvf_loss_high + \
                           self.args.level_weights[1] * dvf_loss_mid + \
                           self.args.level_weights[2] * dvf_loss_low

                loss = ncc_loss + self.args.w_bending_energy * dvf_loss


            # backpropagation
            loss.backward()
            # optimization
            self.optimizer.step()

            # statistics
            epoch_samples += fimage.size(0)
            running_loss += loss.item() * fimage.size(0)
            running_seg_dsc_loss += seg_dsc_loss.item() * fimage.size(0)
            running_reg_dsc_loss += reg_dsc_loss.item() * fimage.size(0)
            running_seg_dsc += (1.0 - seg_dsc_loss_high.item()) * fimage.size(0)
            running_reg_dsc += (1.0 - reg_dsc_loss_high.item()) * fimage.size(0)
            running_ncc_loss += ncc_loss.item() * fimage.size(0)
            running_ncc += (1.0 - ncc_loss_high.item()) * fimage.size(0)
            running_dvf_loss += dvf_loss.item() * fimage.size(0)

            self.data_iteration = (self.current_iteration + 1) * nbatches * wsize
            self.current_iteration += 1

        epoch_loss = running_loss / epoch_samples
        epoch_seg_dsc_loss = running_seg_dsc_loss / epoch_samples
        epoch_reg_dsc_loss = running_reg_dsc_loss / epoch_samples
        epoch_ncc_loss = running_ncc_loss / epoch_samples
        epoch_dvf_loss = running_dvf_loss / epoch_samples
        epoch_seg_dsc = running_seg_dsc / epoch_samples
        epoch_reg_dsc = running_reg_dsc / epoch_samples
        epoch_ncc = running_ncc / epoch_samples

        self.summary_writer.add_scalars("Losses/seg_dsc_loss", {'train': epoch_seg_dsc_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/reg_dsc_loss", {'train': epoch_reg_dsc_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/ncc_loss", {'train': epoch_ncc_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/total_loss", {'train': epoch_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Metrics/seg_dsc", {'train': epoch_seg_dsc}, self.current_epoch)
        self.summary_writer.add_scalars("Metrics/reg_dsc", {'train': epoch_reg_dsc}, self.current_epoch)
        self.summary_writer.add_scalars("Metrics/ncc", {'train': epoch_ncc}, self.current_epoch)
        self.summary_writer.add_scalars("DVF/bending_energy", {'train': epoch_dvf_loss}, self.current_epoch)
        self.summary_writer.add_scalar('number_processed_windows', self.data_iteration, self.current_epoch)


    def validate(self):

        # Set model to evaluation mode
        self.model.eval()
        # initialize stats
        running_loss = 0.
        running_ncc_loss = 0.
        running_dvf_loss = 0.
        running_seg_dsc_loss = 0.
        running_reg_dsc_loss = 0.
        running_seg_dsc = 0.
        running_reg_dsc = 0.
        running_ncc = 0.
        epoch_samples = 0
        i = 1

        with torch.no_grad():

            # Iterate over data
            for batch_idx, (fimage, flabel, mimage, mlabel) in enumerate(self.dataloaders['validation'], 1):
                data_dict = clean_data(fimage, flabel, mimage, mlabel, self.args)
                nbatches, wsize, nchannels, x, y, z, _ = fimage.size()

                # forward pass
                if self.args.network == 'Seg':
                    if self.args.in_channels_seg == 1:
                        res = self.model(data_dict['fimage'])
                    elif self.args.in_channels_seg == 2 and 'Im' in self.args.input_segmentation:
                        res = self.model(data_dict['fimage'], data_dict['mimage'])
                    elif self.args.in_channels_seg == 2 and 'Sm' in self.args.input_segmentation:
                        res = self.model(data_dict['fimage'], data_dict['mlabel'])
                    elif self.args.in_channels_seg == 3:
                        res = self.model(data_dict['fimage'], data_dict['mimage'], data_dict['mlabel'])
                    else:
                        self.logger.error(self.args.input_segmentation, "wrong input")

                    seg_dsc_loss_high, seg_dsc_list_high = self.dsc_loss(data_dict['flabel_high'], res['x_high_res'])
                    seg_dsc_loss_mid, seg_dsc_list_mid = self.dsc_loss(data_dict['flabel_mid'], res['x_mid_res'])
                    seg_dsc_loss_low, seg_dsc_list_low = self.dsc_loss(data_dict['flabel_low'], res['x_low_res'])

                    seg_dsc_loss = self.args.level_weights[0] * seg_dsc_loss_high + \
                                   self.args.level_weights[1] * seg_dsc_loss_mid + \
                                   self.args.level_weights[2] * seg_dsc_loss_low

                    loss = seg_dsc_loss


                elif self.args.network == 'Reg':
                    if self.args.in_channels_reg == 2:
                        res = self.model(data_dict['fimage'], data_dict['mimage'])
                    elif self.args.in_channels_reg == 3:
                        res = self.model(data_dict['fimage'], data_dict['mimage'], data_dict['mlabel'])

                    mimage_high_out = self.spatial_transform(data_dict['mimage_high'], res['high_res_dvf'])
                    mimage_mid_out = self.spatial_transform(data_dict['mimage_mid'], res['mid_res_dvf'])
                    mimage_low_out = self.spatial_transform(data_dict['mimage_low'], res['low_res_dvf'])

                    mlabel_high_out = self.spatial_transform(data_dict['mlabel_high_hot'], res['high_res_dvf'],
                                                             mode='nearest')
                    mlabel_mid_out = self.spatial_transform(data_dict['mlabel_mid_hot'], res['mid_res_dvf'],
                                                            mode='nearest')
                    mlabel_low_out = self.spatial_transform(data_dict['mlabel_low_hot'], res['low_res_dvf'],
                                                            mode='nearest')

                    reg_dsc_loss_high, reg_dsc_list_high = self.dsc_loss(data_dict['flabel_high'], mlabel_high_out,
                                                                         use_activation=False)
                    reg_dsc_loss_mid, reg_dsc_list_mid = self.dsc_loss(data_dict['flabel_mid'], mlabel_mid_out,
                                                                       use_activation=False)
                    reg_dsc_loss_low, reg_dsc_list_low = self.dsc_loss(data_dict['flabel_low'], mlabel_low_out,
                                                                       use_activation=False)

                    ncc_loss_high = self.ncc_loss(data_dict['fimage_high'], mimage_high_out)
                    ncc_loss_mid = self.ncc_loss(data_dict['fimage_mid'], mimage_mid_out)
                    ncc_loss_low = self.ncc_loss(data_dict['fimage_low'], mimage_low_out)

                    dvf_loss_high = self.smooth_loss(res['high_res_dvf'])
                    dvf_loss_mid = self.smooth_loss(res['mid_res_dvf'])
                    dvf_loss_low = self.smooth_loss(res['low_res_dvf'])

                    reg_dsc_loss = self.args.level_weights[0] * reg_dsc_loss_high + \
                                   self.args.level_weights[1] * reg_dsc_loss_mid + \
                                   self.args.level_weights[2] * reg_dsc_loss_low

                    ncc_loss = self.args.level_weights[0] * ncc_loss_high + \
                               self.args.level_weights[1] * ncc_loss_mid + \
                               self.args.level_weights[2] * ncc_loss_low

                    dvf_loss = self.args.level_weights[0] * dvf_loss_high + \
                               self.args.level_weights[1] * dvf_loss_mid + \
                               self.args.level_weights[2] * dvf_loss_low

                    loss = ncc_loss + self.args.w_bending_energy * dvf_loss


                # statistics
                epoch_samples += fimage.size(0)
                running_loss += loss.item() * fimage.size(0)
                running_seg_dsc_loss += seg_dsc_loss.item() * fimage.size(0)
                running_reg_dsc_loss += reg_dsc_loss.item() * fimage.size(0)
                running_seg_dsc += (1.0 - seg_dsc_loss_high.item()) * fimage.size(0)
                running_reg_dsc += (1.0 - reg_dsc_loss_high.item()) * fimage.size(0)
                running_ncc_loss += ncc_loss.item() * fimage.size(0)
                running_ncc += (1.0 - ncc_loss_high.item()) * fimage.size(0)
                running_dvf_loss += dvf_loss.item() * fimage.size(0)

                self.data_iteration = (self.current_iteration + 1) * nbatches * wsize
                self.current_iteration += 1

            epoch_loss = running_loss / epoch_samples
            epoch_seg_dsc_loss = running_seg_dsc_loss / epoch_samples
            epoch_reg_dsc_loss = running_reg_dsc_loss / epoch_samples
            epoch_ncc_loss = running_ncc_loss / epoch_samples
            epoch_dvf_loss = running_dvf_loss / epoch_samples
            epoch_seg_dsc = running_seg_dsc / epoch_samples
            epoch_reg_dsc = running_reg_dsc / epoch_samples
            epoch_ncc = running_ncc / epoch_samples

            self.summary_writer.add_scalars("Losses/seg_dsc_loss", {'validation': epoch_seg_dsc_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/reg_dsc_loss", {'validation': epoch_reg_dsc_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/ncc_loss", {'validation': epoch_ncc_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/total_loss", {'validation': epoch_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Metrics/seg_dsc", {'validation': epoch_seg_dsc}, self.current_epoch)
            self.summary_writer.add_scalars("Metrics/reg_dsc", {'validation': epoch_reg_dsc}, self.current_epoch)
            self.summary_writer.add_scalars("Metrics/ncc", {'validation': epoch_ncc}, self.current_epoch)
            self.summary_writer.add_scalars("DVF/bending_energy", {'validation': epoch_dvf_loss}, self.current_epoch)
            self.summary_writer.add_scalar('number_processed_windows', self.data_iteration, self.current_epoch)

            self.logger.info('{} totalLoss: {:.4f} dscLoss: {:.4f} nccLoss: {:.4f} dvfLoss: {:.4f} dsc: {:.4f} ncc: {:.4f}'.
                    format('validation', epoch_loss, epoch_reg_dsc_loss, epoch_ncc_loss, epoch_dvf_loss, epoch_reg_dsc))


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
        evaluation(self.args, self.config)


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
