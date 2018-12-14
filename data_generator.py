# -*-coding:utf-8-*-
import numpy as np
from vessel_cutter import *
from data_loader import readVesselFromGE3DSavedState, readCCTASeries
import SimpleITK as sitk
import os, nrrd, cv2
import shutil

def generateData(dcm_path, threeDSavedStatePath, data_save_path, dilate_rate=1):
    dcm_name = dcm_path.split('/')[-1]
    print 'processing %s' %(dcm_name)
    try:
        sitk_img, vessels = readCCTASeries(dcm_path, threeDSavedStatePath)
    except:
        print 'Failed to read CCTA series'
        return

    # save raw dcm data as nrrd
    img_save_path = os.path.join(data_save_path, dcm_name)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    else:
        return
    sitk.WriteImage(sitk_img, os.path.join(img_save_path, 'img.nrrd'))

    # generate label as label.nrrd for training, if 3D saved state file doesnt contain either 'Right Coronary Artery',
    # 'Left Circumflex Artery', 'Left Anterior Descending Artery', skip it and delete the corresponding img
    if generateLabel(vessels, sitk_img, nrrd_save_path=img_save_path, dilate_rate=dilate_rate):
        shutil.rmtree(img_save_path)

def generateLabel(vessels, sitk_img, nrrd_save_path, dilate_rate=1):
    vessel_keys = ['Right Coronary Artery','Left Circumflex Artery', 'First Diagonal', 'Left Anterior Descending Artery']
    sitk_img_shape = sitk.GetArrayFromImage(sitk_img).shape
    # get two set of labels, for left and right coronary arteries respectively, their intersection will be labeled as left
    # right: 1
    # left:  2
    label_array = np.zeros((2, sitk_img_shape[0], sitk_img_shape[1], sitk_img_shape[2])).astype('int16')
    for key_idx, vessel_key in enumerate(vessel_keys):
        if vessel_key == 'First Diagonal':
            # some 3D saved state files might not have 'First Diagonal' info, we still take it into account
            try:
                cross_section = getVesselCrossSection(vessels, vessel_key)
                cross_section_thresh = getCrossSectionThresh(cross_section)
                contour_match_list = get3DContourMinMatch(vessels, vessel_key, thresh=cross_section_thresh)
            except:
                continue
        else:
            # if 3D saved state file doesnt contain either 'Right Coronary Artery','Left Circumflex Artery', 'Left Anterior Descending Artery'
            # skip it
            try:
                cross_section = getVesselCrossSection(vessels, vessel_key)
                cross_section_thresh = getCrossSectionThresh(cross_section)
                contour_match_list = get3DContourMinMatch(vessels, vessel_key, thresh=cross_section_thresh)
            except:
                return 1
        for i in range(label_array.shape[1]):
            z = (np.array(sitk_img.GetOrigin()) + np.array(sitk_img.GetDirection()[-3:]) * sitk_img.GetSpacing()[-1] * i)[2]
            contour_point_list = getContourPointList(contour_match_list, z)
            if len(contour_point_list) > 0:
                contour_point_pixel_array = getContourPointPixel(contour_point_list, sitk_img.GetSpacing(),
                                                                 sitk_img.TransformIndexToPhysicalPoint([0, 0, i]))
                for contour_point in contour_point_pixel_array:
                    #print contour_point
                    if key_idx == 0 :
                        label_array[0, i, contour_point[0][0], contour_point[0][1]] = 1
                    else:
                        label_array[1, i, contour_point[0][0], contour_point[0][1]] = 2
    label_array[1][label_array[1] == 2] = 1
    label_array[0] = binaryClosing(label_array[0])
    label_array[1] = binaryClosing(label_array[1])
    # right: 2
    # left: 1
    label_array[1][(label_array[0] == 1) * (label_array[1] == 0)] = 2
    label_sitk_img = sitk.GetImageFromArray(label_array[1].transpose(0, 2, 1).astype('uint8'))
    label_sitk_img.CopyInformation(sitk_img)

    if not os.path.exists(nrrd_save_path):
        os.mkdir(nrrd_save_path)
    sitk.WriteImage(label_sitk_img, os.path.join(nrrd_save_path, 'label.nrrd'))

def binaryClosing(array, dilate_rate=1):
    check_binary(array)
    closed_img = sitk.BinaryMorphologicalClosing(sitk.GetImageFromArray(array), 1)
    return  sitk.GetArrayFromImage(closed_img)

def binaryClosing(array, dilate_rate=2, erode_rate=1):
    check_binary(array)
    closed_img = sitk.BinaryDilate(sitk.GetImageFromArray(array), dilate_rate)
    closed_img = sitk.BinaryErode(closed_img, erode_rate)
    return sitk.GetArrayFromImage(closed_img)

def check_binary(array):
    # print np.histogram(array)
    # print np.sum((array == 1)), np.sum((array == 0)), np.prod(array.shape)
    assert np.sum((array == 1)) + np.sum((array == 0)) == np.prod(array.shape), 'input array is not binary!'

# def getMask(contour_point_list, center_point_list, mask_shape=(512, 512)):
#     if len(center_point_list) <= 1:
#         x, y = np.meshgrid(np.arange(mask_shape[0]), np.arange(mask_shape[1]))
#         x, y = x.flatten, y.flatten
#
#         points = np.vstack((x, y)).T
#
#         mask = points_inside_poly(points, contour_point_list)
#         return mask
#     # if there are multiple center points, cluster contour points according to Euclidean metric and then find out all points
#     # inside each polygon
#     else:
#         polygon_list = [[] for _ in center_point_list]
#         dis_mat = np.zeros((len(contour_point_list), len(center_point_list)))
#         for i, contour_point in enumerate(contour_point_list):
#             for j, center_point in enumerate(center_point_list):
#                 dis_mat[i, j] = getEucDis(np.array(contour_point), np.array(center_point))
#         min_dis_idx = np.argmin(dis_mat, axis=1)
#         for i in min_dis_idx:
#             polygon_list[min_dis_idx[i]].append(contour_point_list[i])
#         mask = np.ones(mask_shape)
#         for point_set in polygon_list:
#             mask * getMask(point_set, [])
#         return mask

def saveMaskedImage(nrrd_path, masked_img_save_path, window_center=300, window_width=800):
    if not os.path.exists(masked_img_save_path):
        os.mkdir(masked_img_save_path)
    img_array = nrrd.read(os.path.join(nrrd_path, 'img.nrrd'))[0].transpose(2, 1, 0)
    label_array = nrrd.read(os.path.join(nrrd_path, 'label.nrrd'))[0].transpose(2, 0, 1)
    for i, (img, label) in enumerate(zip(img_array, label_array)):
        windowed_img = window_convert_light(img, window_center, window_width)
        windowed_img_rgb = windowed_img[..., np.newaxis]
        windowed_img_rgb = np.repeat(windowed_img_rgb, axis=2, repeats=3)
        img_to_draw, _ = contour_and_draw(windowed_img_rgb, label)
        cv2.imwrite(os.path.join(masked_img_save_path, 'img_%d.png' %(i)), img_to_draw)


# draw contour for identified calcified region
def contour_and_draw(image, label_map, n_class=3, shape=(512, 512)):
    #image should be (512,512,3), label_map should be (512, 512)
    all_contours=[]
    for c_id in range(1, n_class):
        one_channel = np.zeros(shape, dtype=np.uint8)
        one_channel[label_map == c_id] = 1
        _, contours, hierarchy = cv2.findContours(one_channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
        if c_id == 1:
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
        elif c_id == 2:
            cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    return image, all_contours

def generateBatchData(dcm_root_path, data_save_path):
    for dcm_name in os.listdir(dcm_root_path):
        dcm_patient_path = os.path.join(dcm_root_path, dcm_name)
        file_list = os.listdir(dcm_patient_path)
        try:
            assert len(file_list) == 2, 'patient folder must contain only one dcm folder and one save state .dcm!'
        except:
            continue
        for file_name in file_list:
            if file_name.endswith('.dcm'):
                threeDSavedStatePath = os.path.join(dcm_patient_path, file_name)
            else:
                dcm_path = os.path.join(dcm_patient_path, file_name)
        generateData(dcm_path, threeDSavedStatePath, data_save_path)



if __name__ == '__main__':
    vessel_keys = ['Left Circumflex Artery', 'First Diagonal', 'Left Anterior Descending Artery',
                   'Right Coronary Artery']
    saved_state_dir = '/media/tx-eva-cc/data/3D_test/3Dsavedstate/BJFW11827750S306/BJFW11827750S306_001.dcm'
    dcm_dir = '/home/tx-eva-cc/Desktop/11025082/11025082_SScoreSerSav'
    nrrd_path = '/media/tx-eva-cc/data/3D_recon_data/test_data/test_data/BJFW11827750'
    masked_img_save_path = '/media/tx-eva-cc/data/3D_recon_data/test_data/test_data/BJFW11827750_masked_img'
    # print (np.array(contour_match_list)).shape
    dcm_path = '/media/tx-eva-cc/data/3D_test/3Dsavedstate/BJFW505889S652/BJFW505889'
    threeDSavedStatePath = '/media/tx-eva-cc/data/3D_test/3Dsavedstate/BJFW505889S652/BJFW505889S652_001.dcm'
    data_save_path = '/media/tx-eva-cc/data/3D_recon_data/test_data/test_data'
    # generateData(dcm_path,
    #              threeDSavedStatePath,
    #              data_save_path, dilate_rate=1)
    # saveMaskedImage(os.path.join(data_save_path, 'BJFW505889'), os.path.join(data_save_path, 'BJFW505889_masked_img_2_1'))
    dcm_root_path = '/media/tx-eva-cc/data/3D_test/3D-Saved-State-AutoSave2'
    data_save_path = '/media/tx-eva-cc/data/3D_recon_data/train_data/FW_third_batch_train/FW_third_batch_train/FW_third_batch_train'
    generateBatchData(dcm_root_path, data_save_path)