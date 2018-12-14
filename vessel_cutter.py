# -*-coding:utf-8-*-
from data_loader import readVesselFromGE3DSavedState

import numpy as np
import pydicom as dicom
import cv2, os
from polygon import polyArea

# containing tools for cutting vessels into its cross section on each CT slice

def window_convert_light(pix, center, width):
    '''Perform intensity windowing on pix array in a memory-saving way, operation equivalent to
    pix = ((pix - center + 0.5) / (width - 1) + 0.5) * 255'''
    new_img = pix.astype('float32')
    new_img -= (center - 0.5)
    new_img /= (width - 1)
    new_img += 0.5
    new_img *= 255
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype('uint8')

def check_point_dim(point):
    # check if the points has dim 3
    assert np.shape(point) == (3, ), 'input point must be a numpy array of shape (3, )'

# get the shortest-distance matching between two planes of points
def getMinMatch(point_list1, point_list2):
    if len(point_list1) == 0 or len(point_list2) == 0:
        return []
    distance_metric = getDisMetric(point_list1, point_list2)
    min_idx = getMinIdx(distance_metric)
    min_match = []
    for idx in min_idx:
        min_match.append((point_list1[idx[0]], point_list2[idx[1]]))
    return np.array(min_match)

# given a distance metric, find the indices of pairs with shorted Euclidean distance
def getMinIdx(distance_metric):
    x_idx_list = list(np.argmax(distance_metric, axis=0))
    y_idx_list = list(np.argmax(distance_metric, axis=1))
    min_idx = []
    for i in range(len(x_idx_list)):
        min_idx.append((x_idx_list[i], i))
    for j in range(len(y_idx_list)):
        min_idx.append((j, y_idx_list[j]))
    min_idx = set(min_idx)
    return min_idx

# get the distance metric (matrix) between two sets of points
def getDisMetric(point_list1, point_list2):
    if len(point_list1) == 0 or len(point_list2) == 0:
        print 'one of the input point lists is empty'
        return
    else:
        x_dim = len(point_list1)
        y_dim = len(point_list2)
        distance_metric = np.zeros((x_dim, y_dim))
        for i in range(x_dim):
            for j in range(y_dim):
                distance_metric[i, j] = getEucDis(point_list1[i], point_list2[j])
        return distance_metric

# get the Euclidean distance of two points
def getEucDis(point1, point2):
    check_point_dim(point1)
    check_point_dim(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))

# get the shortest-distance match for a whole 3D tubes (multiple layers of cross sections)
# e.g. out put =
#  [[[   18.314102  -126.74044   1600.025879]
#  [   19.025238  -127.251022  1600.367065]], ...]
def get3DContourMinMatch(vessels, key, sub_key='contourPoints', thresh=60):
    dict_list =  vessels[key]['contours']
    #print dict_list
    contour_match_list = []
    for i, _ in enumerate(dict_list):
        if i < len(dict_list) - 1 and i > thresh:
            if sub_key == 'centerPoint':
                contour_match_list += list(getMinMatch([dict_list[i][sub_key]], [dict_list[i + 1][sub_key]]))
            elif sub_key == 'contourPoints':
                contour_match_list += list(getMinMatch(dict_list[i][sub_key], dict_list[i+1][sub_key]))
            else:
                NotImplemented
    return np.array(contour_match_list).astype('float32')

def getContourPointList(contour_match_list, z):
    # e.g. matched_pair =
    # [[   18.314102  -126.74044   1600.025879]
    #  [   19.025238  -127.251022  1600.367065]]
    contour_point_list = []
    #print contour_match_list, z
    for matched_pair in contour_match_list:
        matched_pair_max = max(matched_pair[0, 2], matched_pair[1, 2])
        matched_pair_min = min(matched_pair[0, 2], matched_pair[1, 2])
        if z >= matched_pair_min and z <= matched_pair_max:
            x_pair = matched_pair[:, 0]
            y_pair = matched_pair[:, 1]
            z_pair = matched_pair[:, 2]
            x_interpolated = (x_pair[0] * (z_pair[1] - z) + x_pair[1] * (z - z_pair[0])) / (z_pair[1] - z_pair[0])
            y_interpolated = (y_pair[0] * (z_pair[1] - z) + y_pair[1] * (z - z_pair[0])) / (z_pair[1] - z_pair[0])
            contour_point_list.append([x_interpolated, y_interpolated])
    return contour_point_list

# calculate cross section area of the vessel (area enclosed by the points of the contour), sampled for each centerline point
def getVesselCrossSection(vessels, key):
    dict_list = vessels[key]['contours']
    area_list = []
    for i, dict in enumerate(dict_list):
        point_list = list(dict_list[i]['contourPoints'])
        area = polyArea(point_list)
        area_list.append(area)

    return area_list

# get the corresponding coordinate projection of the contour point onto each slice of the CT scan
def getContourPointPixel(contour_point_list, PixelSpacing, ImagePositionPatient):
    if len(contour_point_list) == 0:
        return
    contour_point_pixel_list = []
    for contour_point in contour_point_list:
        x = round((contour_point[0] - ImagePositionPatient[0]) / PixelSpacing[0])
        y = round((contour_point[1] - ImagePositionPatient[1]) / PixelSpacing[1])
        contour_point_pixel_list.append([x, y])
    contour_point_pixel_array = (np.array(contour_point_pixel_list)).reshape((-1, 1, 2)).astype(np.int32)
    return contour_point_pixel_array

# draw contour on each slice of the CT scan
def drawContour(dcm_path, img_save_path, contour_match_list):
    dcm_list = os.listdir(dcm_path)
    for dcm_name in dcm_list:
        dcm_file_path = os.path.join(dcm_path, dcm_name)
        dcm = dicom.read_file(dcm_file_path)
        windowed_img_rgb = getWindowedImgRgb(dcm)
        for i, match_list in enumerate(contour_match_list):
            #######
            # we take all z's to be absolute values to avoid sign problems
            contour_point_list = getContourPointList(match_list, z=abs(dcm.SliceLocation))
            contour_point_pixel_array = getContourPointPixel(contour_point_list, dcm.PixelSpacing, dcm.ImagePositionPatient)
            # if contour is not empty
            #print contour_point_pixel_array
            if contour_point_pixel_array is not None:
                if i > 0:
                    cv2.drawContours(windowed_img_rgb, [contour_point_pixel_array], 0, (0, 0, 255), 1)
                else:
                    for contour_point in contour_point_pixel_array:
                        cv2.drawContours(windowed_img_rgb, [contour_point], 0, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(img_save_path, (dcm_file_path.split('/')[-1]).split('.')[0] + '.png'), windowed_img_rgb)

# from a dcm (pydicom.dataset.FileDataset) generated windowed RGB image
def getWindowedImgRgb(dcm, window_center=300, window_width=800):
    array = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    windowed_img = window_convert_light(array, window_center, window_width)
    windowed_img_rgb = windowed_img[..., np.newaxis]
    windowed_img_rgb = np.repeat(windowed_img_rgb, axis=2, repeats=3)
    return windowed_img_rgb

def getCrossSectionThresh(cross_sections, ratio_thresh=0.5):
    max_cross_section = max(cross_sections)
    for i, cross_section in enumerate(cross_sections):
        if cross_section < max_cross_section * ratio_thresh:
            return i

if __name__ == '__main__':
    vessel_keys = ['Left Circumflex Artery', 'First Diagonal', 'Left Anterior Descending Artery', 'Right Coronary Artery']
    saved_state_dir = '/media/tx-eva-cc/data/3D_test/3Dsavedstate/BJFW11826357S552/BJFW11826357S552_001.dcm'
    dcm_dir = '/home/tx-eva-cc/Desktop/11025082/11025082_SScoreSerSav'
    vessels = readVesselFromGE3DSavedState(saved_state_dir)
    #print vessels.keys()
    cross_section = getVesselCrossSection(vessels, vessel_keys[0])
    cross_section_thresh = getCrossSectionThresh(cross_section)
    #print cross_section_thresh
    contour_match_list = get3DContourMinMatch(vessels, vessel_keys[0], thresh=cross_section_thresh)
    center_match_list = get3DContourMinMatch(vessels, vessel_keys[0], sub_key='centerPoint', thresh=cross_section_thresh)
    # print (np.array(contour_match_list)).shape
    drawContour('/media/tx-eva-cc/data/3D_test/3Dsavedstate/BJFW11826357S552/BJFW11826357', '/media/tx-eva-cc/data/3D_test/test/test_show/BJFW11826357_LCA', [contour_match_list, center_match_list])
