import logging
import os
import time

import SimpleITK as sitk
import numpy as np
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
from utils.sliding_window_inference import SlidingWindow
from utils.util import clean_data, resize_image_mlvl


class stlAgent(BaseAgent):
    def __init__(self, args, data_config):
        super(stlAgent).__init__()

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
                                    channels_list=self.args.num_featurmaps).to(self.args.device)
                # Create instance from the loss
                # self.dsc_loss = Multi_DSC_Loss().to(self.args.device)
                self.dsc_loss = Dice().to(self.args.device)
            elif self.args.network == 'Reg':
                self.model = RegNet(in_channels=len(self.args.input), dim=self.args.num_classes,
                                    depth=self.args.depth, initial_channels=self.args.initial_channels,
                                    channels_list=self.args.num_featurmaps).to(self.args.device)
                # Create instance from the loss
                self.ncc_loss = NCC(3, self.args.ncc_window_size).to(self.args.device)
                self.smooth_loss = GradientSmoothing(energy_type='bending')
                self.dsc_loss = Dice().to(self.args.device)
                self.spatial_transform = SpatialTransformer(dim=self.args.num_classes)
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

            if (epoch) % self.args.validation_rate == 0:
                self.validate()

            self.save_checkpoint()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def train_one_epoch(self):

        # initialize stats
        running_loss = 0.
        running_ncc_loss = 0.
        running_smooth_loss = 0.
        running_dsc_loss = 0.
        running_dsc = 0.
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
            if len(self.args.input) == 1:
                res = self.model(data_dict['fimage'])
            elif len(self.args.input) == 2 and 'Im' in self.args.input:
                res = self.model(data_dict['fimage'], data_dict['mimage'])
            elif len(self.args.input) and 'Sm' in self.args.input:
                res = self.model(data_dict['fimage'], data_dict['mlabel'])
            elif len(self.args.input) == 3:
                res = self.model(data_dict['fimage'], data_dict['mimage'], data_dict['mlabel'])
            else:
                self.logger.error(self.args.input_segmentation, "wrong input")

            if self.args.network == 'Seg':
                dsc_loss_high, dsc_list_high = self.dsc_loss(data_dict['flabel_high'], res['logits_high'])
                dsc_loss_mid, dsc_list_mid = self.dsc_loss(data_dict['flabel_mid'], res['logits_mid'])
                dsc_loss_low, dsc_list_low = self.dsc_loss(data_dict['flabel_low'], res['logits_low'])

                dsc_loss = self.args.level_weights[0] * dsc_loss_high + \
                               self.args.level_weights[1] * dsc_loss_mid + \
                               self.args.level_weights[2] * dsc_loss_low

                loss = dsc_loss


            elif self.args.network == 'Reg':

                mimage_high_out = self.spatial_transform(data_dict['mimage_high'], res['dvf_high'])
                mimage_mid_out = self.spatial_transform(data_dict['mimage_mid'], res['dvf_mid'])
                mimage_low_out = self.spatial_transform(data_dict['mimage_low'], res['dvf_low'])

                mlabel_high_out = self.spatial_transform(data_dict['mlabel_high_hot'], res['dvf_high'],
                                                         mode='nearest')
                mlabel_mid_out = self.spatial_transform(data_dict['mlabel_mid_hot'], res['dvf_mid'], mode='nearest')
                mlabel_low_out = self.spatial_transform(data_dict['mlabel_low_hot'], res['dvf_low'], mode='nearest')

                reg_dsc_loss_high, reg_dsc_list_high = self.dsc_loss(data_dict['flabel_high'], mlabel_high_out,
                                                                     use_activation=False)
                reg_dsc_loss_mid, reg_dsc_list_mid = self.dsc_loss(data_dict['flabel_mid'], mlabel_mid_out,
                                                                   use_activation=False)
                reg_dsc_loss_low, reg_dsc_list_low = self.dsc_loss(data_dict['flabel_low'], mlabel_low_out,
                                                                   use_activation=False)

                ncc_loss_high = self.ncc_loss(data_dict['fimage_high'], mimage_high_out)
                ncc_loss_mid = self.ncc_loss(data_dict['fimage_mid'], mimage_mid_out)
                ncc_loss_low = self.ncc_loss(data_dict['fimage_low'], mimage_low_out)

                smooth_loss_high = self.smooth_loss(res['dvf_high'])
                smooth_loss_mid = self.smooth_loss(res['dvf_mid'])
                smooth_loss_low = self.smooth_loss(res['dvf_low'])

                dsc_loss = self.args.level_weights[0] * reg_dsc_loss_high + \
                               self.args.level_weights[1] * reg_dsc_loss_mid + \
                               self.args.level_weights[2] * reg_dsc_loss_low

                ncc_loss = self.args.level_weights[0] * ncc_loss_high + \
                           self.args.level_weights[1] * ncc_loss_mid + \
                           self.args.level_weights[2] * ncc_loss_low

                smooth_loss = self.args.level_weights[0] * smooth_loss_high + \
                           self.args.level_weights[1] * smooth_loss_mid + \
                           self.args.level_weights[2] * smooth_loss_low

                loss = ncc_loss + self.args.w_bending_energy * smooth_loss

            # backpropagation
            loss.backward()
            # optimization
            self.optimizer.step()

            # statistics
            epoch_samples += fimage.size(0)
            running_loss += loss.item() * fimage.size(0)
            running_dsc_loss += dsc_loss.item() * fimage.size(0)
            running_dsc += (1.0 - dsc_loss.item()) * fimage.size(0)
            if self.args.network == 'Reg':
                running_ncc_loss += ncc_loss.item() * fimage.size(0)
                running_ncc += (1.0 - ncc_loss.item()) * fimage.size(0)
                running_smooth_loss += smooth_loss.item() * fimage.size(0)

            self.data_iteration = (self.current_iteration + 1) * nbatches * wsize
            self.current_iteration += 1

        epoch_loss = running_loss / epoch_samples
        epoch_dsc_loss = running_dsc_loss / epoch_samples
        epoch_dsc = running_dsc / epoch_samples
        self.summary_writer.add_scalars("Losses/dsc_loss", {'train': epoch_dsc_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/total_loss", {'train': epoch_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Metrics/dsc", {'train': epoch_dsc}, self.current_epoch)
        self.summary_writer.add_scalar('number_processed_windows', self.data_iteration, self.current_epoch)

        if self.args.network == 'Seg':
            self.logger.info('{} totalLoss: {:.4f} dscLoss: {:.4f} dsc: {:.4f}'.
                             format('Training', epoch_loss, epoch_dsc_loss, epoch_dsc))
        elif self.args.network == 'Reg':
            epoch_ncc_loss = running_ncc_loss / epoch_samples
            epoch_smooth_loss = running_smooth_loss / epoch_samples
            epoch_ncc = running_ncc / epoch_samples
            self.summary_writer.add_scalars("Losses/ncc_loss", {'train': epoch_ncc_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Metrics/ncc", {'train': epoch_ncc}, self.current_epoch)
            self.summary_writer.add_scalars("DVF/bending_energy", {'train': epoch_smooth_loss}, self.current_epoch)
            self.logger.info('{} totalLoss: {:.4f} dscLoss: {:.4f} nccLoss: {:.4f} dvfLoss: {:.4f} dsc: {:.4f} ncc: {:.4f}'.
                format('Training', epoch_loss, epoch_dsc_loss, epoch_ncc_loss, epoch_smooth_loss, epoch_dsc, epoch_ncc))



    def validate(self):

        # Set model to evaluation mode
        self.model.eval()
        # initialize stats
        running_loss = 0.
        running_ncc_loss = 0.
        running_smooth_loss = 0.
        running_dsc_loss = 0.
        running_dsc = 0.
        running_ncc = 0.
        epoch_samples = 0

        with torch.no_grad():

            # Iterate over data
            for batch_idx, (fimage, flabel, mimage, mlabel) in enumerate(self.dataloaders['validation'], 1):
                data_dict = clean_data(fimage, flabel, mimage, mlabel, self.args)
                nbatches, wsize, nchannels, x, y, z, _ = fimage.size()

                # forward pass
                if len(self.args.input) == 1:
                    res = self.model(data_dict['fimage'])
                elif len(self.args.input) == 2 and 'Im' in self.args.input:
                    res = self.model(data_dict['fimage'], data_dict['mimage'])
                elif len(self.args.input) == 2 and 'Sm' in self.args.input:
                    res = self.model(data_dict['fimage'], data_dict['mlabel'])
                elif len(self.args.input) == 3:
                    res = self.model(data_dict['fimage'], data_dict['mimage'], data_dict['mlabel'])
                else:
                    self.logger.error(self.args.input, "wrong input")

                if self.args.network == 'Seg':
                    dsc_loss_high, dsc_list_high = self.dsc_loss(data_dict['flabel_high'], res['logits_high'])
                    dsc_loss_mid, dsc_list_mid = self.dsc_loss(data_dict['flabel_mid'], res['logits_mid'])
                    dsc_loss_low, dsc_list_low = self.dsc_loss(data_dict['flabel_low'], res['logits_low'])

                    dsc_loss = self.args.level_weights[0] * dsc_loss_high + \
                               self.args.level_weights[1] * dsc_loss_mid + \
                               self.args.level_weights[2] * dsc_loss_low

                    loss = dsc_loss


                elif self.args.network == 'Reg':

                    mimage_high_out = self.spatial_transform(data_dict['mimage_high'], res['dvf_high'])
                    mimage_mid_out = self.spatial_transform(data_dict['mimage_mid'], res['dvf_mid'])
                    mimage_low_out = self.spatial_transform(data_dict['mimage_low'], res['dvf_low'])

                    mlabel_high_out = self.spatial_transform(data_dict['mlabel_high_hot'], res['dvf_high'],
                                                             mode='nearest')
                    mlabel_mid_out = self.spatial_transform(data_dict['mlabel_mid_hot'], res['dvf_mid'],
                                                            mode='nearest')
                    mlabel_low_out = self.spatial_transform(data_dict['mlabel_low_hot'], res['dvf_low'],
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

                    smooth_loss_high = self.smooth_loss(res['dvf_high'])
                    smooth_loss_mid = self.smooth_loss(res['dvf_mid'])
                    smooth_loss_low = self.smooth_loss(res['dvf_low'])

                    dsc_loss = self.args.level_weights[0] * reg_dsc_loss_high + \
                               self.args.level_weights[1] * reg_dsc_loss_mid + \
                               self.args.level_weights[2] * reg_dsc_loss_low

                    ncc_loss = self.args.level_weights[0] * ncc_loss_high + \
                               self.args.level_weights[1] * ncc_loss_mid + \
                               self.args.level_weights[2] * ncc_loss_low

                    smooth_loss = self.args.level_weights[0] * smooth_loss_high + \
                                  self.args.level_weights[1] * smooth_loss_mid + \
                                  self.args.level_weights[2] * smooth_loss_low

                    loss = ncc_loss + self.args.w_bending_energy * smooth_loss


                # statistics
                epoch_samples += fimage.size(0)
                running_loss += loss.item() * fimage.size(0)
                running_dsc_loss += dsc_loss.item() * fimage.size(0)
                running_dsc += (1.0 - dsc_loss.item()) * fimage.size(0)
                if self.args.network == 'Reg':
                    running_ncc_loss += ncc_loss.item() * fimage.size(0)
                    running_ncc += (1.0 - ncc_loss.item()) * fimage.size(0)
                    running_smooth_loss += smooth_loss.item() * fimage.size(0)

                self.data_iteration = (self.current_iteration + 1) * nbatches * wsize
                self.current_iteration += 1

            epoch_loss = running_loss / epoch_samples
            epoch_dsc_loss = running_dsc_loss / epoch_samples
            epoch_dsc = running_dsc / epoch_samples
            self.summary_writer.add_scalars("Losses/dsc_loss", {'train': epoch_dsc_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Losses/total_loss", {'train': epoch_loss}, self.current_epoch)
            self.summary_writer.add_scalars("Metrics/dsc", {'train': epoch_dsc}, self.current_epoch)
            self.summary_writer.add_scalar('number_processed_windows', self.data_iteration, self.current_epoch)

            if self.args.network == 'Seg':
                self.logger.info('{} totalLoss: {:.4f} dscLoss: {:.4f} dsc: {:.4f}'.
                                 format('validation', epoch_loss, epoch_dsc_loss, epoch_dsc))
            elif self.args.network == 'Reg':
                epoch_ncc_loss = running_ncc_loss / epoch_samples
                epoch_smooth_loss = running_smooth_loss / epoch_samples
                epoch_ncc = running_ncc / epoch_samples
                self.summary_writer.add_scalars("Losses/ncc_loss", {'train': epoch_ncc_loss}, self.current_epoch)
                self.summary_writer.add_scalars("Metrics/ncc", {'train': epoch_ncc}, self.current_epoch)
                self.summary_writer.add_scalars("DVF/bending_energy", {'train': epoch_smooth_loss}, self.current_epoch)

                self.logger.info('{} totalLoss: {:.4f} dscLoss: {:.4f} nccLoss: {:.4f} dvfLoss: {:.4f} dsc: {:.4f} ncc: {:.4f}'.
                    format('validation', epoch_loss, epoch_dsc_loss, epoch_ncc_loss, epoch_smooth_loss, epoch_dsc, epoch_ncc))

    def inference(self):

        for partition in self.args.split_set:
            if partition == 'validation':
                dataset = 'HMC'
            elif partition == 'inference':
                dataset = 'EMC'

            reader = self.dsets[partition].sampler.reader
            inference_cases = reader._file_list['fixed_image'].values

            pl_bshape = self.args.patch_size  # 96 x 96 x 96
            op_shape = [l1 - l2 for l1, l2 in zip(self.args.patch_size, self.args.out_diff_size)]# 1x1x56x56x56
            out_diff = np.array(pl_bshape) - np.array(op_shape)  # 40x40x40
            padding = [[0, 0]]  + [[diff // 2, diff - diff // 2] for diff in out_diff] + [[0, 0]]
            self.model.to(self.args.device)
            self.model.eval()

            for i in range(len(inference_cases)):

                print(inference_cases[i].split('/')[-4], inference_cases[i].split('/')[-3])

                _, data, _ = reader(idx=i, shuffle=False)

                fimage = data['fixed_image'][..., 0, :]
                mimage = data['moving_image'][..., 0, :]
                mlabel = data['moving_segmentation'][..., 0, :]

                fimage[fimage > 1000] = 1000
                fimage[fimage < -1000] = -1000
                fimage = fimage / 1000

                mimage[mimage > 1000] = 1000
                mimage[mimage < -1000] = -1000
                mimage = mimage / 1000

                fimage = np.expand_dims(fimage, axis=0)
                mimage = np.expand_dims(mimage, axis=0)
                mlabel = np.expand_dims(mlabel, axis=0)

                inp_shape = fimage.shape  # 1x1x122x512x512
                inp_bshape = inp_shape[1:-1]  # 122x512x512
                fimage_padded = np.pad(fimage, padding, mode='constant', constant_values=fimage.min())
                mimage_padded = np.pad(mimage, padding, mode='constant', constant_values=fimage.min())
                mlabel_padded = np.pad(mlabel, padding, mode='constant', constant_values=fimage.min())
                f_bshape = fimage_padded.shape[1:-1]  # 162x552x552
                striding = (list(np.maximum(1, np.array(op_shape) // 2)) if all(out_diff == 0) else op_shape)
                out_fimage_dummies = np.zeros(inp_shape)  # 1x1x122x512x512
                out_label_dummies = np.zeros(inp_shape)  # 1x1x122x512x512
                out_dvf_dummies = np.zeros([inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3], 3])  # 1x1x122x512x512x3

                sw = SlidingWindow(f_bshape, pl_bshape, striding=striding)
                out_sw = SlidingWindow(inp_bshape, op_shape, striding=striding)
                done = False

                while True:
                    try:
                        slicer = next(sw)
                        out_slicer = next(out_sw)
                    except StopIteration:
                        done = True

                    fimage_window= fimage_padded[tuple(slicer)]
                    mimage_window = mimage_padded[tuple(slicer)]
                    mlabel_window = mlabel_padded[tuple(slicer)]

                    fimage_window = torch.tensor(np.transpose(fimage_window, (0, 4, 1, 2, 3))).to(self.args.device) #BxCxDxWxH
                    mimage_window = torch.tensor(np.transpose(mimage_window, (0, 4, 1, 2, 3))).to(self.args.device)  # BxCxDxWxH
                    mlabel_window = torch.tensor(np.transpose(mlabel_window, (0, 4, 1, 2, 3))).to(self.args.device)  # BxCxDxWxH

                    with torch.no_grad():
                        # forward pass
                        if self.args.in_channels == 1:
                            res = self.model(fimage_window)
                        elif self.args.in_channels == 2 and 'Im' in self.args.input:
                            res = self.model(fimage_window, mimage_window)
                        elif self.args.in_channels == 2 and 'Sm' in self.args.input:
                            res = self.model(fimage_window, mimage_window)
                        elif self.args.in_channels == 3:
                            res = self.model(fimage_window, mimage_window, mlabel_window)
                        else:
                            self.logger.error(self.args.input, "wrong input")

                    if self.args.network == 'Seg':
                        probs = F.softmax(res['logits_high'], dim=1)
                        _, segmentation = torch.max(probs, dim=1, keepdim=True)
                        out_label_dummies[tuple(out_slicer)] = np.transpose(segmentation.cpu().numpy(), (0, 2, 3, 4, 1)) #BxDxWxHxC

                    elif self.args.network == 'Reg':
                        mimage_window_high = resize_image_mlvl(self.args, mimage_window, 0)
                        mlabel_window_high = resize_image_mlvl(self.args, mlabel_window, 0)
                        mimage_high_out = self.spatial_transform(mimage_window_high, res['dvf_high'], mode='trilinear')
                        mlabel_high_out = self.spatial_transform(mlabel_window_high, res['dvf_high'], mode='nearest')
                        out_fimage_dummies[tuple(out_slicer)] = np.transpose(mimage_high_out.cpu().numpy(), (0, 2, 3, 4, 1)) #BxDxWxHxC
                        out_dvf_dummies[tuple(out_slicer)] = res['dvf_high'].cpu().numpy() #BxDxWxHxC
                        out_label_dummies[tuple(out_slicer)] = np.transpose(mlabel_high_out.cpu().numpy(),
                                                                             (0, 2, 3, 4, 1))  # BxDxWxHxC

                    if done:
                        break

                save_dir = os.path.join(self.args.prediction_dir, dataset, inference_cases[i].split('/')[-4],
                                        inference_cases[i].split('/')[-3])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                out_fimage_dummies = out_fimage_dummies * 1000
                im_itk = sitk.ReadImage(inference_cases[i])

                flabel_itk = sitk.GetImageFromArray(np.squeeze(out_label_dummies.astype(np.uint8)))
                flabel_itk.SetOrigin(im_itk.GetOrigin())
                flabel_itk.SetSpacing([1., 1., 1.])
                # flabel_itk.SetSpacing(im_itk.GetSpacing())
                flabel_itk.SetDirection(im_itk.GetDirection())

                if self.args.network == 'Reg':
                    sitk.WriteImage(flabel_itk, os.path.join(save_dir, 'Segmentation.mha'))

                elif self.args.network == 'Reg':

                    fimage_itk = sitk.GetImageFromArray(np.squeeze(out_fimage_dummies.astype(np.int16)))
                    fimage_itk.SetOrigin(im_itk.GetOrigin())
                    fimage_itk.SetSpacing([1., 1., 1.])
                    # fimage_itk.SetSpacing(im_itk.GetSpacing())
                    fimage_itk.SetDirection(im_itk.GetDirection())

                    dvf_itk = sitk.GetImageFromArray(np.squeeze(out_dvf_dummies), isVector=True)
                    dvf_itk.SetOrigin(im_itk.GetOrigin())
                    dvf_itk.SetSpacing([1., 1., 1.])
                    # dvf_itk.SetSpacing(im_itk.GetSpacing())
                    dvf_itk.SetDirection(im_itk.GetDirection())

                    sitk.WriteImage(flabel_itk, os.path.join(save_dir, 'ResampledSegmentation.mha'))
                    sitk.WriteImage(fimage_itk, os.path.join(save_dir, 'ResampledImage.mha'))
                    sitk.WriteImage(dvf_itk, os.path.join(save_dir, 'DVF.mha'))



    def eval(self):
        evaluation(self.args, self.data_config)

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
