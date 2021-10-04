import math
import time
from math import floor, sqrt

import SimpleITK as sitk
import numpy as np


def modified_LC(maskImg, comp=1):
    # initialize the connected component filter
    ccFilter = sitk.ConnectedComponentImageFilter()
    # apply the filter to the input image
    labelImg = ccFilter.Execute(maskImg)
    # get the number of labels (connected components)
    numberOfLabels = ccFilter.GetObjectCount()
    # extract the data array from the itk object
    labelArray = sitk.GetArrayFromImage(labelImg)
    # count the voxels belong to different components
    labelSizes = np.bincount(labelArray.flatten())
    labelSizes2 = np.asarray(sorted(labelSizes, reverse=True))
    # get the largest connected component
    # largestLabel = np.argmax(labelSizes[4:]) + 1

    if(comp>=labelSizes2.shape[0]):
        return None
    else:
        largestLabel = np.argwhere(labelSizes == labelSizes2[comp])[0][0]
        # convert the data array to itk object
        outImg = sitk.GetImageFromArray((labelArray == largestLabel).astype(np.int16))
        # output image should have same metadata as input mask image
        outImg.CopyInformation(maskImg)

        return outImg

def idx_to_coor(n, k):
    i = floor((-sqrt((2*n+1)*(2*n+1)-8*k)+2*n+1)/2)
    j = k + i - i*(2*n-i+1)//2
    return i, j

def modified_LC_SV(maskImg, numComps, idx):
    if (idx <= numComps):
        predicted_image_lc = modified_LC(maskImg, idx)
    else:
        i, j = idx_to_coor(numComps, idx-1)
        j += 1
        predicted_image_lc_i = modified_LC(maskImg, i)
        predicted_image_lc_j = modified_LC(maskImg, j)
        predicted_image_lc = predicted_image_lc_i + predicted_image_lc_j
    predicted_image_lc = sitk.Cast(predicted_image_lc, sitk.sitkUInt8)
    return predicted_image_lc

def resampler_sitk_Seg(image_sitk, predicted_image, spacing=[1.0, 1.0, 1.0], default_pixel_value=0,
                   interpolator=sitk.sitkNearestNeighbor, dimension=3, rnd=5):
    ratio = [round(spacing_dim / spacing[i], 6) for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
    ImRef = sitk.Image(tuple(math.ceil(size_dim * ratio[i]) for i, size_dim in enumerate(image_sitk.GetSize())),
                       sitk.sitkInt16)
    #ImRef.SetOrigin(image_sitk.GetOrigin())
    ImRef.SetOrigin(predicted_image.GetOrigin())
    ImRef.SetDirection(image_sitk.GetDirection())
    ImRef.SetSpacing(spacing)
    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ImRef)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(identity)
    resampled_sitk = resampler.Execute(image_sitk)

    return resampled_sitk

def resampler_sitk_Reg(image_sitk, spacing=[1.0, 1.0, 1.0], default_pixel_value=0,
                   interpolator=sitk.sitkNearestNeighbor, dimension=3, rnd=3):
    ratio = [spacing_dim / spacing[i] for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
    ImRef = sitk.Image(tuple(math.ceil(size_dim * ratio[i]) for i, size_dim in enumerate(image_sitk.GetSize())),
                       sitk.sitkInt16)
    ImRef.SetOrigin(image_sitk.GetOrigin())
    ImRef.SetDirection(image_sitk.GetDirection())
    ImRef.SetSpacing(spacing)
    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ImRef)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(identity)
    resampled_sitk = resampler.Execute(image_sitk)

    return resampled_sitk


def DSC_MSD_HD95_Seg(groundtruth_image_itk, predicted_image, num_of_components,resample_flag=True, resample_spacing=[1.0, 1.0, 1.0]):
    if resample_flag:
        groundtruth_image_itk = resampler_sitk_Seg(image_sitk=groundtruth_image_itk, predicted_image=predicted_image,
                                               spacing=resample_spacing,
                                               default_pixel_value=0,
                                               interpolator=sitk.sitkNearestNeighbor, dimension=3, rnd=3)


    groundtruth_image_itk = sitk.Cast(groundtruth_image_itk, sitk.sitkUInt8)
    # predicted_image = sitk.Cast(predicted_image, sitk.sitkUInt8)
    size_diff = np.sum(np.subtract(groundtruth_image_itk.GetSize(), predicted_image.GetSize()))

    if size_diff > 0:
        if size_diff == 1:
            groundtruth_image_itk = groundtruth_image_itk[:, :, :-1]
        if size_diff == 2:
            groundtruth_image_itk = groundtruth_image_itk[:-1, :-1, :]
        elif size_diff == 3:
            groundtruth_image_itk = groundtruth_image_itk[:-1, :-1, :-1]
        else:
            print(size_diff)

    elif size_diff < 0:
        if size_diff == -2:
            predicted_image = predicted_image[:-1, :-1, :]
        elif size_diff == -3:
            predicted_image = predicted_image[:-1, :-1, :-1]
        else:
            print(size_diff)

    else:
        pass

    label_overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
    dsc_test = []

    startTime = 0

    isSV = (num_of_components > 10)  # TODO: this should be a function parameter instead.
    for i in range(1, 50):  # MAX NUM COMPONENTS TO CONSIDER
        if i == num_of_components:
            startTime = time.time()
        predicted_image_lc = modified_LC(predicted_image, i)
        if predicted_image_lc is None:
            break
        predicted_image_lc = sitk.Cast(predicted_image_lc, sitk.sitkUInt8)
        try:
            label_overlap_measures_filter.Execute(groundtruth_image_itk, predicted_image_lc)
        except:
            pass
        val = label_overlap_measures_filter.GetDiceCoefficient()
        dsc_test.append(val)
        if val > 0.5 and not isSV:
            break

    numComps = len(dsc_test)

    #TODO: loop ranges could be until numComps, that would be faster than num_of_components
    #TODO: I guess you could ignore a component if it has DSC 0.0, but that'd not be worth the effort to code.

    if isSV and numComps > 1:
        if startTime == 0:
            startTime = time.time()
        for i in range(1, 50):  # MAX NUM COMPONENTS TO CONSIDER FOR COMBINED SV
            predicted_image_lc_i = modified_LC(predicted_image, i)
            if predicted_image_lc_i is None:
                break
            # print("  SV " + repr(i) + ", dsc " + repr(dsc_test[i-1]))
            for j in range(i + 1, 50):  # MAX NUM COMPONENTS TO CONSIDER FOR COMBINED SV
                predicted_image_lc_j = modified_LC(predicted_image, j)
                if predicted_image_lc_j is None:
                    break
                predicted_image_lc = predicted_image_lc_i + predicted_image_lc_j
                predicted_image_lc = sitk.Cast(predicted_image_lc, sitk.sitkUInt8)
                try:
                    label_overlap_measures_filter.Execute(groundtruth_image_itk, predicted_image_lc)
                except:
                    pass
                val = label_overlap_measures_filter.GetDiceCoefficient()
                dsc_test.append(val)
        assert(len(dsc_test) == (numComps+1)*numComps/2)

    if startTime != 0:
        endTime = time.time()
        # print("Extra time: " + repr(endTime - startTime) + ".")
    else:
        # print("No time added.")
        pass

    if predicted_image_lc is None and i == 1: #Needed to prevent crash if the network did not predict an organ at all
        print("No predicted pixels for this organ...!")
        predicted_image[0, 0, 0] = 1
        predicted_image = sitk.Cast(predicted_image, sitk.sitkUInt8)
        dsc_test = []
    else:
        LC = np.argmax(np.asarray(dsc_test)) + 1
        dsc_test = []
        predicted_image = modified_LC_SV(predicted_image, numComps, LC)

    label_overlap_measures_filter.Execute(groundtruth_image_itk, predicted_image)
    dice = label_overlap_measures_filter.GetDiceCoefficient()
    # print("    dsc = " + repr(dice))
    jaccard = label_overlap_measures_filter.GetJaccardCoefficient()
    vol_similarity = label_overlap_measures_filter.GetVolumeSimilarity()

    hausdorff_distance_image_filter.Execute(groundtruth_image_itk, predicted_image)

    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(groundtruth_image_itk, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(groundtruth_image_itk)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    segmented_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(predicted_image, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(predicted_image)

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))

    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

    all_surface_distances = seg2ref_distances + ref2seg_distances
    msd = np.mean(all_surface_distances)
    hd_percentile = np.maximum(np.percentile(seg2ref_distances, 95), np.percentile(ref2seg_distances, 95))

    return dice, msd, hd_percentile, jaccard, vol_similarity


def DSC_MSD_HD95_Reg(groundtruth_image_itk, predicted_image, resample_flag=True, resample_spacing=[1.0, 1.0, 1.0]):

    if resample_flag:
        groundtruth_image_itk = resampler_sitk_Reg(image_sitk=groundtruth_image_itk, spacing=resample_spacing,
                                                default_pixel_value=0,
                                                interpolator=sitk.sitkNearestNeighbor, dimension=3, rnd=3)

    groundtruth_image_itk = sitk.Cast(groundtruth_image_itk, sitk.sitkUInt8)
    predicted_image = sitk.Cast(predicted_image, sitk.sitkUInt8)
    size_diff = np.sum(np.subtract(groundtruth_image_itk.GetSize(), predicted_image.GetSize()))

    if size_diff > 0:
        if size_diff == 2:
            groundtruth_image_itk = groundtruth_image_itk[:-1, :-1, :]
        elif size_diff == 2:
            groundtruth_image_itk = groundtruth_image_itk[:-1, :-1, :-1]
        elif size_diff == 1:
            groundtruth_image_itk = groundtruth_image_itk[:, :, :-1]
        else:
            print(size_diff)
    elif size_diff < 0:
        if size_diff == -2:
            predicted_image = predicted_image[:-1, :-1, :]
        elif size_diff == -3:
            predicted_image = predicted_image[:-1, :-1, :-1]
        elif size_diff == -1:
            predicted_image = predicted_image[:, :, :-1]
        else:
            print(size_diff)
    else:
        pass

    try:

        label_overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        label_overlap_measures_filter.Execute(groundtruth_image_itk, predicted_image)

        dice = label_overlap_measures_filter.GetDiceCoefficient()
        jaccard = label_overlap_measures_filter.GetJaccardCoefficient()
        vol_similarity = label_overlap_measures_filter.GetVolumeSimilarity()

        hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_image_filter.Execute(groundtruth_image_itk, predicted_image)

        reference_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(groundtruth_image_itk, squaredDistance=False, useImageSpacing=True))
        reference_surface = sitk.LabelContour(groundtruth_image_itk)

        statistics_image_filter = sitk.StatisticsImageFilter()
        # Get the number of pixels in the reference surface by counting all pixels that are 1.
        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())

        segmented_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(predicted_image, squaredDistance=False, useImageSpacing=True))
        segmented_surface = sitk.LabelContour(predicted_image)

        # Multiply the binary surface segmentations with the distance maps. The resulting distance
        # maps contain non-zero values only on the surface (they can also contain zero on the surface)
        seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
        ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

        # Get the number of pixels in the reference surface by counting all pixels that are 1.
        statistics_image_filter.Execute(segmented_surface)
        num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

        # Get all non-zero distances and then add zero distances if required.
        seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
        seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
        seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))

        ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
        ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
        ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

        all_surface_distances = seg2ref_distances + ref2seg_distances
        msd = np.mean(all_surface_distances)
        hd_percentile = np.maximum(np.percentile(seg2ref_distances, 95), np.percentile(ref2seg_distances, 95))

        return dice, msd, hd_percentile, jaccard, vol_similarity
    except:
        print("AN EXCEPTION OCCURRED IN LABEL_EVAL.PY!")
        return 0, 0, 0, 0, 0