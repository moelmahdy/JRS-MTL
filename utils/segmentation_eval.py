import os

import SimpleITK as sitk
import pandas as pd
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner

from utils.dataset_niftynet import set_dataParam
from utils.overlap_quantification import DSC_MSD_HD95_Seg, DSC_MSD_HD95_Reg


class evaluation(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.partitioner = ImageSetsPartitioner().initialise(data_param=set_dataParam(self.args, self.config),
                                                             data_split_file=self.config.csv_split_file,
                                                             new_partition=False)
        self.run()

    def run(self):

        for partition in self.args.split_set:
            if partition == 'validation':
                dataset = 'HMC'
            elif partition == 'inference':
                dataset = 'EMC'

            files_list = self.partitioner.get_file_list(partition, 'fixed_segmentation')['fixed_segmentation'].values.tolist()



            for j in range(len(self.args.label_name)):

                writer = pd.ExcelWriter(os.path.join(self.args.prediction_dir, dataset, self.args.excel_name[j]),
                                        engine='xlsxwriter')

                # refresh the values
                self.dsc_bladder = []
                self.msd_bladder = []
                self.hd_bladder = []

                self.dsc_rectum = []
                self.msd_rectum = []
                self.hd_rectum = []

                self.dsc_gtv = []
                self.msd_gtv = []
                self.hd_gtv = []

                self.dsc_sv = []
                self.msd_sv = []
                self.hd_sv = []

                self.patient_arr = []
                self.scan_arr = []

                if os.path.exists(os.path.join(self.args.prediction_dir, dataset, self.args.excel_name[j])):
                    pass
                else:

                    for file in files_list:
                        patient_name = file.split('/')[-4]
                        visit_name = file.split('/')[-3]

                        groundtruth_segmentation = sitk.ReadImage(file)
                        groundtruth_bladder = sitk.BinaryThreshold(groundtruth_segmentation, lowerThreshold=1,
                                                                   upperThreshold=1, insideValue=1, outsideValue=0)
                        groundtruth_rectum = sitk.BinaryThreshold(groundtruth_segmentation, lowerThreshold=2,
                                                                  upperThreshold=2, insideValue=1, outsideValue=0)
                        groundtruth_sv = sitk.BinaryThreshold(groundtruth_segmentation, lowerThreshold=3,
                                                              upperThreshold=3, insideValue=1, outsideValue=0)
                        groundtruth_prostate = sitk.BinaryThreshold(groundtruth_segmentation, lowerThreshold=4,
                                                                    upperThreshold=4, insideValue=1, outsideValue=0)

                        predicted_segmentation = sitk.ReadImage((os.path.join(self.args.prediction_dir, dataset,
                                                                              patient_name, visit_name, self.args.label_name[j])))


                        predicted_bladder = sitk.BinaryThreshold(predicted_segmentation, lowerThreshold=1,
                                                                 upperThreshold=1, insideValue=1, outsideValue=0)
                        predicted_rectum = sitk.BinaryThreshold(predicted_segmentation, lowerThreshold=2,
                                                                upperThreshold=2, insideValue=1, outsideValue=0)
                        predicted_sv = sitk.BinaryThreshold(predicted_segmentation, lowerThreshold=3,
                                                            upperThreshold=3, insideValue=1, outsideValue=0)
                        predicted_prostate = sitk.BinaryThreshold(predicted_segmentation, lowerThreshold=4,
                                                                  upperThreshold=4, insideValue=1, outsideValue=0)


                        if 'Resampled' in self.args.label_name[j]:
                            dsc_bladder, msd_bladder, hd_bladder, _, _ = DSC_MSD_HD95_Reg(groundtruth_bladder, predicted_bladder,
                                                                                      resample_spacing=self.args.voxel_dim)
                            dsc_rectum, msd_rectum, hd_rectum, _, _ = DSC_MSD_HD95_Reg(groundtruth_rectum, predicted_rectum,
                                                                                   resample_spacing=self.args.voxel_dim)
                            dsc_sv, msd_sv, hd_sv, _, _ = DSC_MSD_HD95_Reg(groundtruth_sv, predicted_sv,
                                                                       resample_spacing=self.args.voxel_dim)
                            dsc_prostate, msd_prostate, hd_prostate, _, _ = DSC_MSD_HD95_Reg(groundtruth_prostate, predicted_prostate,
                                                                                         resample_spacing=self.args.voxel_dim)

                        else:
                            dsc_bladder, msd_bladder, hd_bladder, _, _ = DSC_MSD_HD95_Seg(groundtruth_bladder,
                                                                                          predicted_bladder,
                                                                                          self.args.num_components[1],
                                                                                          resample_spacing=self.args.voxel_dim)
                            dsc_rectum, msd_rectum, hd_rectum, _, _ = DSC_MSD_HD95_Seg(groundtruth_rectum,
                                                                                       predicted_rectum,
                                                                                       self.args.num_components[1],
                                                                                       resample_spacing=self.args.voxel_dim)
                            dsc_sv, msd_sv, hd_sv, _, _ = DSC_MSD_HD95_Seg(groundtruth_sv, predicted_sv,
                                                                           self.args.num_components[0],
                                                                           resample_spacing=self.args.voxel_dim)
                            dsc_prostate, msd_prostate, hd_prostate, _, _ = DSC_MSD_HD95_Seg(groundtruth_prostate,
                                                                                             predicted_prostate,
                                                                                             self.args.num_components[
                                                                                                 1],
                                                                                             resample_spacing=self.args.voxel_dim)

                        self.patient_arr.append(patient_name)
                        self.scan_arr.append(visit_name)

                        self.dsc_bladder.append(dsc_bladder)
                        self.msd_bladder.append(msd_bladder)
                        self.hd_bladder.append(hd_bladder)

                        self.dsc_rectum.append(dsc_rectum)
                        self.msd_rectum.append(msd_rectum)
                        self.hd_rectum.append(hd_rectum)

                        self.dsc_gtv.append(dsc_prostate)
                        self.msd_gtv.append(msd_prostate)
                        self.hd_gtv.append(hd_prostate)

                        self.dsc_sv.append(dsc_sv)
                        self.msd_sv.append(msd_sv)
                        self.hd_sv.append(hd_sv)

                        print(dataset, patient_name, visit_name)


                    data = {'Patient': self.patient_arr, 'Scan': self.scan_arr,
                            'DSC_Bladder': self.dsc_bladder, 'MSD_Bladder': self.msd_bladder,
                            'HD%95_Bladder': self.hd_bladder,'DSC_Rectum': self.dsc_rectum,
                            'MSD_Rectum': self.msd_rectum, 'HD%95_Rectum': self.hd_rectum,
                            'DSC_SeminalVesicle': self.dsc_sv, 'MSD_SeminalVesicle': self.msd_sv,
                            'HD%95_SeminalVesicle': self.hd_sv, 'DSC_GTV': self.dsc_gtv,
                            'MSD_GTV': self.msd_gtv, 'HD%95_GTV': self.hd_gtv}

                    df = pd.DataFrame(data, dtype=float)

                    df = df.reindex(['Patient', 'Scan', 'DSC_Bladder', 'MSD_Bladder', 'HD%95_Bladder','DSC_Rectum', 'MSD_Rectum',
                                     'HD%95_Rectum','DSC_SeminalVesicle', 'MSD_SeminalVesicle', 'HD%95_SeminalVesicle','DSC_GTV', 'MSD_GTV', 'HD%95_GTV'], axis=1)

                    df.loc['Median'] = df.median()
                    df.loc['Min'] = df.min()
                    df.loc['Max'] = df.max()
                    df.loc['Q75'] = df.quantile(.75)
                    df.loc['Q25'] = df.quantile(.25)
                    df.loc['IQR'] = df.loc['Q75'] - df.loc['Q25']
                    df.loc['LowerOutlierLimit'] = df.loc['Q25'] - (df.loc['IQR'] * 1.5)
                    df.loc['UpperOutlierLimit'] = df.loc['Q75'] + (df.loc['IQR'] * 1.5)
                    df.loc['Mean'] = df.mean()
                    df.loc['Std'] = df.std()

                    df.to_excel(writer, sheet_name='eval')
                    writer.save()