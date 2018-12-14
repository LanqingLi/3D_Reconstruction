# -*-coding:utf-8-*-
# requires pydicom >= 1.2.0
# requires cssselect

import pydicom
from StringIO import StringIO
from lxml import etree
import numpy as np
import pandas as pd
import SimpleITK as sitk
import glob
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib


# Extract vessel info (centerline and vessel cross section contour) from GE 3D Saved State
def readVesselFromGE3DSavedState(fname):
    savedState = pydicom.dcmread(fname, specific_tags=[pydicom.tag.TupleTag((0x57, 0x1005))])
    if type(savedState[0x00571005].value) is pydicom.sequence.Sequence:
        savedTree = etree.fromstring(savedState[0x00571005][0][0x00571044].value)
    elif isinstance(savedState[0x00571005].value, basestring):
        fp = StringIO(savedState[0x57, 0x1005].value)
        savedTree = etree.fromstring(
            pydicom.filereader.read_sequence(fp, True, True, len(savedState[0x57, 0x1005].value), "utf-8")[0][
                0x00571044].value)

    vesselsRawData = {i.find("name").text:
                          {'rawData': i,
                           'interpolatedPoints': np.array(
                               [[- float(j.attrib['x']), - float(j.attrib['y']), float(j.attrib['z'])] for j in
                                i.cssselect("points>external_coords>point")], dtype=np.float),
                           'originalPoints': np.array(
                               [[- float(j.attrib['x']), - float(j.attrib['y']), float(j.attrib['z'])] for j in
                                i.cssselect("originalPaths>point")], dtype=np.float)}
                      for i in savedTree.cssselect("leave")}
    for i in vesselsRawData.keys():
        res = []
        for j in vesselsRawData[i]['rawData'].cssselect('points>external_coords>point'):
            cur = {
                'centerPoint': np.array([- float(j.attrib['x']), - float(j.attrib['y']), (float(j.attrib['z']))]).astype(
                    'float'),
                'contourPoints': np.array(
                    [[- float(k.attrib['x']), - float(k.attrib['y']), (float(k.attrib['z']))] for k in
                     j.cssselect('point>floatContour>contour')]).astype('float')
            }
            res.append(cur)
        vesselsRawData[i]['contours'] = res
    return vesselsRawData

# find the corresponding series ID of the GE 3D Saved State Info
def readSeriesIdFromGE3DSavedState(fname):
    savedState = pydicom.dcmread(fname, specific_tags=[pydicom.tag.TupleTag((0x57, 0x1005))])
    if type(savedState[0x00571005].value) is pydicom.sequence.Sequence:
        series_instance_id = etree.fromstring(savedState[0x00571005][0][0x0020000e].value)
    elif isinstance(savedState[0x00571005].value, basestring):
        fp = StringIO(savedState[0x57, 0x1005].value)

        series_instance_id = pydicom.filereader.read_sequence(fp, True, True, len(savedState[0x57, 0x1005].value), "utf-8")[0][0x0020000e]
        print series_instance_id

    return series_instance_id

# 用法
# fname = "/home/tx-eva-32/wr/fuwai_cta/ctca/1.2.528.1.1001.200.10.1229.4269.1.20170908045918134/SDY00000/SRS00007/IMG00000.DCM"
# readVesselFromGE3DSavedState(fname)
# 里面你们需要用于重建和分析的点是interpolatedPoints
# contour信息可以自己parse rawData

def readCCTASeriesFromRawDirectory(inPath):
    dcmList = [(i, pydicom.read_file(i, stop_before_pixels=True)) for i in glob.glob(inPath + "/*.dcm")]
    if len(dcmList) < 10:
        raise ValueError("Too few files in directory ended with .dcm.")
    dcmDict = dict(dcmList)
    df = pd.DataFrame({"fname": [i[0] for i in dcmList],
                       'seriesDescription': [i[1].SeriesDescription if "SeriesDescription" in i[1] else "" for i in
                                             dcmList],
                       'seriesNumber': [i[1].SeriesNumber for i in dcmList],
                       'seriesDate': [str(i[1].SeriesDate) if "SeriesDate" in i[1] else "" for i in dcmList],
                       'seriesTime': [str(i[1].SeriesTime) if "SeriesTime" in i[1] else "" for i in dcmList]
                       })
    print df
    df["seriesDateTime"] = pd.to_datetime(df.seriesDate + ' ' + df.seriesTime, unit='ns')
    threeDCandidate = df.query('seriesDescription.str.contains("3D")')
    threeDFname = threeDCandidate.fname.loc[threeDCandidate.seriesDateTime.idxmax()]
    vessels = readVesselFromGE3DSavedState(threeDFname)
    candidateSeriesNames = df[['seriesDescription', 'seriesNumber']].drop_duplicates().query(
        'seriesDescription.str.contains("Smart Phase")')
    if len(candidateSeriesNames) > 0:
        reader = sitk.ImageSeriesReader()
        fnameLst = df[df.seriesNumber == candidateSeriesNames.seriesNumber.iloc[0]].fname.tolist()
        fnameLst.sort(key=lambda x: dcmDict[x].ImagePositionPatient[-1])
        reader.SetFileNames(fnameLst)
        im = reader.Execute()
    else:
        raise Exception('No proper image series found.')
    return im, vessels


def readCCTASeries(cctaImagePath, threeDSavedStatePath):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(cctaImagePath))
    im = reader.Execute()
    vessels = readVesselFromGE3DSavedState(threeDSavedStatePath)
    return im, vessels

def read3DInterpolatedPointsLength(vessels):
    keys = vessels.keys()
    for key in keys:
        print vessels[key]['interpolatedPoints']
        length = len(vessels[key]['interpolatedPoints'])
        print '%s has %d interpolated points' %(key, length)

def read3DOriginalPointsLength(vessels):
    keys = vessels.keys()
    for key in keys:
        print vessels[key]['originalPoints']
        length = len(vessels[key]['originalPoints'])
        print '%s has %d original points' % (key, length)

def read3DContourPointsLength(vessels):
    keys = vessels.keys()
    for key in keys:
        print vessels[key]['contours']
        length = len(vessels[key]['contours'])
        print '%s has %d contour points' % (key, length)

def get3DContourPointsList(vessels, key):
    dict_list =  vessels[key]['contours']
    contour_point_list = []
    for i, dict in enumerate(dict_list):
        if i <= 50:
            contour_point_list += list(dict['contourPoints'])
    return contour_point_list

def plot3DScatter(x_list, y_list, z_list):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_list, y_list, z_list)
    plt.show()

def drawSurfacePlt(point_list):
    point_array = np.array(point_list)
    x = point_array[:, 0]
    y = point_array[:, 1]
    z = point_array[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    saved_state_dir = '/home/tx-eva-cc/Desktop/11025082/11025082_StateAutoSave/BJFW11025082S552_001.dcm'
    dcm_dir = '/home/tx-eva-cc/Desktop/11025082/11025082_SScoreSerSav'
    vessels = readVesselFromGE3DSavedState(saved_state_dir)
    # dcm_im, dcm_vessels = readCCTASeriesFromRawDirectory(dcm_dir)
    # print dcm_im, dcm_vessels
    # vessels is a dict with keys = ['Branch 8', 'Branch 9', 'Right Coronary Artery', 'Branch 3', 'Branch 6', 'Branch 7', 'Left Circumflex Artery', 'First Diagonal', 'Left Anterior Descending Artery']
    #print vessels.keys()
    #print vessels['First Diagonal']['interpolatedPoints']
    #print dcm_im, dcm_vessels
    # series = readSeriesIdFromGE3DSavedState(saved_state_dir)
    # print series
    # read3DContourPointsLength(vessels)
    # polygon_coordinates1 = [(   52.15889 ,  -183.361053,  1586.325806),
    #    (   52.002029,  -183.34108 ,  1586.341187),
    #    (   51.800125,  -183.497925,  1586.228149),
    #    (   51.679031,  -183.616226,  1586.1427  ),
    #    (   51.653587,  -183.765213,  1586.034546),
    #    (   51.600143,  -184.00415 ,  1585.860962),
    #    (   51.927742,  -184.223984,  1585.699463),
    #    (   51.970398,  -184.229568,  1585.69519 ),
    #    (   52.196831,  -184.157196,  1585.746582),
    #    (   52.369614,  -184.044098,  1585.828003),
    #    (   52.472542,  -183.872253,  1585.952393),
    #    (   52.41835 ,  -183.738815,  1586.049683),
    #    (   52.414551,  -183.602631,  1586.148804),
    #    (   52.292984,  -183.492294,  1586.229614)]
    # polygon_coordinates2 = [[   52.40876 ,  -183.389725,  1586.93689 ],
    #    [   52.143188,  -183.361359,  1586.970337],
    #    [   51.883614,  -183.387726,  1586.961182],
    #    [   51.659508,  -183.51033 ,  1586.876221],
    #    [   51.502106,  -183.61142 ,  1586.804932],
    #    [   51.338387,  -183.783966,  1586.678711],
    #    [   51.181816,  -184.001953,  1586.517212],
    #    [   51.176094,  -184.210297,  1586.356445],
    #    [   51.301521,  -184.390747,  1586.21167 ],
    #    [   51.513306,  -184.485138,  1586.129517],
    #    [   51.759201,  -184.515472,  1586.095459],
    #    [   52.098442,  -184.500259,  1586.092529],
    #    [   52.315491,  -184.499954,  1586.083252],
    #    [   52.619354,  -184.382645,  1586.160767],
    #    [   52.790848,  -184.238586,  1586.264526],
    #    [   52.845612,  -184.011871,  1586.437378],
    #    [   52.805977,  -183.746689,  1586.643921],
    #    [   52.657982,  -183.519531,  1586.825806]]
    # x1 = [i[0] for i in polygon_coordinates1]
    # y1 = [i[1] for i in polygon_coordinates1]
    # z1 = [i[2] for i in polygon_coordinates1]
    # x2 = [i[0] for i in polygon_coordinates2]
    # y2 = [i[1] for i in polygon_coordinates2]
    # z2 = [i[2] for i in polygon_coordinates2]
    # plot3DScatter(x1 + x2, y1 + y2, z1 + z2)
    contour_point_list = get3DContourPointsList(vessels, 'Left Circumflex Artery')
    drawSurfacePlt(contour_point_list)
    (np.source(Axes3D.plot_trisurf))