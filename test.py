# -*-coding:utf-8-*-
import vtk, pyevtk
import numpy as np
import vtk
from IPython.display import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

from mayavi import mlab

def drawSurfaceDelaunay(point_list):
    x = [i[0] for i in point_list]
    y = [i[1] for i in point_list]
    z = [i[2] for i in point_list]

    pts = mlab.points3d(x, y, z, z)

    mesh = mlab.pipeline.delaunay2d(pts)

    pts.remove()

    surf = mlab.pipeline.surface(mesh)

    mlab.xlabel('x')
    mlab.ylabel('y')
    mlab.zlabel('z')
    mlab.show()

def drawSurfacePlt(point_list):
    point_array = np.array(point_list)
    x = point_array[:, 0]
    y = point_array[:, 1]
    z = point_array[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(x - x.mean(), y - y.mean(), z - z.mean(), cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    plt.show()

# def drawSurface(point_list):
#     ellip_base = ellipsoid(6, 10, 16, levelset=True)
#     print ellip_base.shape
#     # Use marching cubes to obtain the surface mesh of these ellipsoids
#     point_array = np.array(point_list)
#     verts, faces, normals, values = measure.marching_cubes_lewiner(point_array, 0)
#
#     # Display resulting triangular mesh using Matplotlib. This can also be done
#     # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Fancy indexing: `verts[faces]` to generate a collection of triangles
#     mesh = Poly3DCollection(verts[faces])
#     mesh.set_edgecolor('k')
#     ax.add_collection3d(mesh)
#
#     ax.set_xlabel("x-axis: a = 6 per ellipsoid")
#     ax.set_ylabel("y-axis: b = 10")
#     ax.set_zlabel("z-axis: c = 16")
#
#     ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
#     ax.set_ylim(0, 20)  # b = 10
#     ax.set_zlim(0, 32)  # c = 16
#
#     plt.tight_layout()
#     plt.show()

# Makes a vtkIdList from a Python iterable. I'm kinda surprised that
# this is necessary, since I assumed that this kind of thing would
# have been built into the wrapper and happen transparently, but it
# seems not.
# def mkVtkIdList(it):
#     vil = vtk.vtkIdList()
#     for i in it:
#         vil.InsertNextId(int(i))
#     return vil
#
#
# def getVtkPoints(point_list):
#     points = vtk.vtkPoints()
#     for i, point in enumerate(point_list):
#         points.InsertPoint(i, point)
#     return points
#
# def getSurface(vtk_points):
#     dmc = vtk.vtkDiscreteMarchingCubes()
#     data_object = vtk.vtkPolyData()
#     data_object.SetPoints(vtk_points)
#     dmc.SetInput(data_object)
#     dmc.GenerateValues(1, 1, 1)
#     dmc.Update()
#
#
# def vtk_show(renderer, width=400, height=300):
#     """
#     Takes vtkRenderer instance and returns an IPython Image with the rendering.
#     """
#     renderWindow = vtk.vtkRenderWindow()
#     renderWindow.SetOffScreenRendering(1)
#     renderWindow.AddRenderer(renderer)
#     renderWindow.SetSize(width, height)
#     renderWindow.Render()
#
#     windowToImageFilter = vtk.vtkWindowToImageFilter()
#     windowToImageFilter.SetInput(renderWindow)
#     windowToImageFilter.Update()
#
#     writer = vtk.vtkPNGWriter()
#     writer.SetWriteToMemory(1)
#     writer.SetInputConnection(windowToImageFilter.GetOutputPort())
#     writer.Write()
#     data = str(buffer(writer.GetResult()))
#
#     return Image(data)
#
# def visualization(vtkDiscreteMarchingCubes):
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputConnection(vtkDiscreteMarchingCubes.GetOutputPort())
#
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#
#     renderer = vtk.vtkRenderer()
#     renderer.AddActor(actor)
#     renderer.SetBackground(1.0, 1.0, 1.0)
#
#     camera = renderer.MakeCamera()
#     camera.SetPosition(-500.0, 245.5, 122.0)
#     camera.SetFocalPoint(301.0, 245.5, 122.0)
#     camera.SetViewAngle(30.0)
#     camera.SetRoll(-90.0)
#     renderer.SetActiveCamera(camera)
#     vtk_show(renderer, 600, 600)


if __name__ == '__main__':
    polygon_coordinates1 = [(52.15889, -183.361053, 1586.325806),
                                (52.002029, -183.34108, 1586.341187),
                                (51.800125, -183.497925, 1586.228149),
                                (51.679031, -183.616226, 1586.1427),
                                (51.653587, -183.765213, 1586.034546),
                                (51.600143, -184.00415, 1585.860962),
                                (51.927742, -184.223984, 1585.699463),
                                (51.970398, -184.229568, 1585.69519),
                                (52.196831, -184.157196, 1585.746582),
                                (52.369614, -184.044098, 1585.828003),
                                (52.472542, -183.872253, 1585.952393),
                                (52.41835, -183.738815, 1586.049683),
                                (52.414551, -183.602631, 1586.148804),
                                (52.292984, -183.492294, 1586.229614)]
    polygon_coordinates2 = [[52.40876, -183.389725, 1586.93689],
                            [52.143188, -183.361359, 1586.970337],
                            [51.883614, -183.387726, 1586.961182],
                            [51.659508, -183.51033, 1586.876221],
                            [51.502106, -183.61142, 1586.804932],
                            [51.338387, -183.783966, 1586.678711],
                            [51.181816, -184.001953, 1586.517212],
                            [51.176094, -184.210297, 1586.356445],
                            [51.301521, -184.390747, 1586.21167],
                            [51.513306, -184.485138, 1586.129517],
                            [51.759201, -184.515472, 1586.095459],
                            [52.098442, -184.500259, 1586.092529],
                            [52.315491, -184.499954, 1586.083252],
                            [52.619354, -184.382645, 1586.160767],
                            [52.790848, -184.238586, 1586.264526],
                            [52.845612, -184.011871, 1586.437378],
                            [52.805977, -183.746689, 1586.643921],
                            [52.657982, -183.519531, 1586.825806]]
    polygon_coordinates3 = [[   52.48217 ,  -184.011688,  1587.703979],
       [   52.154022,  -184.03772 ,  1587.743774],
       [   51.840714,  -184.140549,  1587.716309],
       [   51.559052,  -184.31604 ,  1587.62207 ],
       [   51.26017 ,  -184.518051,  1587.508789],
       [   51.093262,  -184.637039,  1587.44043 ],
       [   50.997459,  -184.832672,  1587.294434],
       [   51.092148,  -185.052597,  1587.092407],
       [   51.22599 ,  -185.235748,  1586.913818],
       [   51.453659,  -185.368393,  1586.759888],
       [   51.69191 ,  -185.45755 ,  1586.640503],
       [   51.935905,  -185.434311,  1586.614258],
       [   52.278198,  -185.394226,  1586.583618],
       [   52.632607,  -185.218155,  1586.664673],
       [   52.804306,  -185.037827,  1586.783569],
       [   52.921143,  -184.885437,  1586.889404],
       [   53.033203,  -184.646637,  1587.068481],
       [   53.019318,  -184.437714,  1587.246216],
       [   52.852394,  -184.204437,  1587.473022],
       [   52.598373,  -184.077911,  1587.626709]]
    polygon_coordinates4 = [[   52.406532,  -183.750137,  1587.295044],
       [   52.140396,  -183.652267,  1587.402588],
       [   51.880341,  -183.67981 ,  1587.407227],
       [   51.613251,  -183.804001,  1587.33374 ],
       [   51.453712,  -183.931152,  1587.246704],
       [   51.275543,  -184.08522 ,  1587.139648],
       [   51.076836,  -184.316162,  1586.971924],
       [   51.081787,  -184.561035,  1586.771729],
       [   51.257507,  -184.769394,  1586.583496],
       [   51.458656,  -184.910095,  1586.447632],
       [   51.785088,  -185.062454,  1586.289307],
       [   51.876068,  -185.099625,  1586.249512],
       [   52.195168,  -185.057983,  1586.250244],
       [   52.609993,  -184.897598,  1586.337769],
       [   52.721642,  -184.752487,  1586.444458],
       [   52.900345,  -184.563293,  1586.5802  ],
       [   52.993546,  -184.360733,  1586.735718],
       [   52.993454,  -184.145935,  1586.910889],
       [   52.78582 ,  -183.962128,  1587.08252 ],
       [   52.60704 ,  -183.840973,  1587.199951]]

    point_list = polygon_coordinates1 + polygon_coordinates2 + polygon_coordinates3 + polygon_coordinates4
    # points = getVtkPoints(point_list)
    # surface = getSurface(points)
    # visualization(surface)
    drawSurfacePlt(point_list)
# def main():
#     # polygon_coordinates = array of 3-tuples of float representing the vertices of a cube:
#     polygon_coordinates1 = [(52.15889, -183.361053, 1586.325806),
#                             (52.002029, -183.34108, 1586.341187),
#                             (51.800125, -183.497925, 1586.228149),
#                             (51.679031, -183.616226, 1586.1427),
#                             (51.653587, -183.765213, 1586.034546),
#                             (51.600143, -184.00415, 1585.860962),
#                             (51.927742, -184.223984, 1585.699463),
#                             (51.970398, -184.229568, 1585.69519),
#                             (52.196831, -184.157196, 1585.746582),
#                             (52.369614, -184.044098, 1585.828003),
#                             (52.472542, -183.872253, 1585.952393),
#                             (52.41835, -183.738815, 1586.049683),
#                             (52.414551, -183.602631, 1586.148804),
#                             (52.292984, -183.492294, 1586.229614)]
#     polygon_coordinates2 = [[52.40876, -183.389725, 1586.93689],
#                             [52.143188, -183.361359, 1586.970337],
#                             [51.883614, -183.387726, 1586.961182],
#                             [51.659508, -183.51033, 1586.876221],
#                             [51.502106, -183.61142, 1586.804932],
#                             [51.338387, -183.783966, 1586.678711],
#                             [51.181816, -184.001953, 1586.517212],
#                             [51.176094, -184.210297, 1586.356445],
#                             [51.301521, -184.390747, 1586.21167],
#                             [51.513306, -184.485138, 1586.129517],
#                             [51.759201, -184.515472, 1586.095459],
#                             [52.098442, -184.500259, 1586.092529],
#                             [52.315491, -184.499954, 1586.083252],
#                             [52.619354, -184.382645, 1586.160767],
#                             [52.790848, -184.238586, 1586.264526],
#                             [52.845612, -184.011871, 1586.437378],
#                             [52.805977, -183.746689, 1586.643921],
#                             [52.657982, -183.519531, 1586.825806]]
#
#     # pts = array of 6 4-tuples of vtkIdType (int) representing the faces
#     #     of the cube in terms of the above vertices
#     pts = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
#            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
#
#     # We'll create the building blocks of polydata including data attributes.
#     cube = vtk.vtkPolyData()
#     points = vtk.vtkPoints()
#     polys = vtk.vtkCellArray()
#     scalars = vtk.vtkFloatArray()
#
#     # Load the point, cell, and data attributes.
#     for i in range(8):
#         points.InsertPoint(i, x[i])
#     for i in range(6):
#         polys.InsertNextCell(mkVtkIdList(pts[i]))
#     for i in range(8):
#         scalars.InsertTuple1(i, i)
#
#     # We now assign the pieces to the vtkPolyData.
#     cube.SetPoints(points)
#     del points
#     cube.SetPolys(polys)
#     del polys
#     cube.GetPointData().SetScalars(scalars)
#     del scalars
#
#     # Now we'll look at it.
#     cubeMapper = vtk.vtkPolyDataMapper()
#     if vtk.VTK_MAJOR_VERSION <= 5:
#         cubeMapper.SetInput(cube)
#     else:
#         cubeMapper.SetInputData(cube)
#     cubeMapper.SetScalarRange(0, 7)
#     cubeActor = vtk.vtkActor()
#     cubeActor.SetMapper(cubeMapper)
#
#     # The usual rendering stuff.
#     camera = vtk.vtkCamera()
#     camera.SetPosition(1, 1, 1)
#     camera.SetFocalPoint(0, 0, 0)
#
#     renderer = vtk.vtkRenderer()
#     renWin = vtk.vtkRenderWindow()
#     renWin.AddRenderer(renderer)
#
#     iren = vtk.vtkRenderWindowInteractor()
#     iren.SetRenderWindow(renWin)
#
#     renderer.AddActor(cubeActor)
#     renderer.SetActiveCamera(camera)
#     renderer.ResetCamera()
#     renderer.SetBackground(1, 1, 1)
#
#     renWin.SetSize(300, 300)
#
#     # interact with data
#     renWin.Render()
#     iren.Start()
#
#     # Clean up
#     del cube
#     del cubeMapper
#     del cubeActor
#     del camera
#     del renderer
#     del renWin
#     del iren