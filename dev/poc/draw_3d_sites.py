from geowatch.geoannots.geomodels import SiteModel
import kwimage
# exterior_pts = poly.exterior.data
# Points need xyz
# faceess are cell array

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2  # NOQA
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    # vtkPolygon
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def triangulate_polygon_interior(polygon):
    """
    References:
        https://gis.stackexchange.com/questions/316697/delaunay-triangulation-algorithm-in-shapely-producing-erratic-result
    """
    import numpy as np
    # from shapely.geometry import Polygon
    # import shapely.wkt
    # from shapely.ops import triangulate
    import geopandas as gpd
    from geovoronoi import voronoi_regions_from_coords
    from scipy.spatial import Delaunay
    import kwimage

    poly_points = []
    # gdf_poly_exterior = gpd.GeoDataFrame({'geometry': [polygon.buffer(-0.0000001).exterior]}).explode(index_parts=True).reset_index()
    gdf_poly_exterior = gpd.GeoDataFrame({'geometry': [polygon.exterior]}).explode(index_parts=True).reset_index()
    for geom in gdf_poly_exterior.geometry:
        poly_points += np.array(geom.coords).tolist()

    try:
        polygon.interiors[0]
    except Exception:
        poly_points = poly_points
    else:
        gdf_poly_interior = gpd.GeoDataFrame({'geometry': list(polygon.interiors)}).explode(index_parts=True).reset_index()
        for geom in gdf_poly_interior.geometry:
            poly_points += np.array(geom.coords).tolist()

    poly_points = np.array([item for sublist in poly_points for item in sublist]).reshape(-1, 2)

    poly_shapes, pts = voronoi_regions_from_coords(poly_points, polygon)
    gdf_poly_voronoi = gpd.GeoDataFrame({'geometry': poly_shapes}).explode(index_parts=True).reset_index()

    final_points_accum = []
    final_simplices_accum = []
    index_offset = 0
    # tri_geom = []
    for geom in gdf_poly_voronoi.geometry:
        geom_exterior = np.vstack(geom.exterior.xy).T
        tri = Delaunay(geom_exterior)

        inside_row_indexes = []
        for row_index, simplex_idxs in enumerate(tri.simplices):
            centroid = kwimage.Polygon(exterior=tri.points[simplex_idxs]).to_shapely().centroid
            if centroid.within(polygon):
                inside_row_indexes.append(row_index)

        if len(inside_row_indexes):
            inside_simplicies = tri.simplices[inside_row_indexes]
            final_points_accum.append(tri.points)
            final_simplices_accum.append(inside_simplicies + index_offset)
            index_offset += len(geom_exterior)

        # inside_triangles = [tri for tri in triangulate(geom) if tri.centroid.within(polygon)]
        # tri_geom += inside_triangles
    final_points = np.concatenate(final_points_accum, axis=0)
    final_simplicies = np.concatenate(final_simplices_accum, axis=0)
    return final_points, final_simplicies


def vtk_triangulate_polygon(polygon):
    """
    https://discourse.vtk.org/t/how-to-triangulate-polygon-with-holes/11497

    Seealso:
        https://examples.vtk.org/site/Cxx/Filtering/ConstrainedDelaunay2D/
    """
    import vtk
    import numpy as np
    from vtk.util import numpy_support
    numpy_to_vtk = numpy_support.numpy_to_vtk
    # polygon = kwimage.Polygon.random(n_holes=1, convex=False)
    poly = kwimage.Polygon.coerce(polygon)

    # append_data = vtk.vtkAppendPolyData()
    # append_data.AddInputData(poly.exterior.data)
    # append_data

    xyz_ring = np.array([[x, y, 1] for x, y in poly.exterior.data])

    poly.exterior
    # for interior in poly.interiors:
    #     ...
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz_ring))

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(poly_data)
    triangulator.Update()
    triangle_poly_data = triangulator.GetOutput()

    numpy_support.vtk_to_numpy(triangle_poly_data.GetPointData())

    cell_data = triangle_poly_data.GetCellData()


def main():
    colors = vtkNamedColors()

    # # Setup four points
    # points = vtkPoints()
    # points.InsertNextPoint(0.0, 0.0, 0.0)
    # points.InsertNextPoint(1.0, 0.0, 0.0)
    # points.InsertNextPoint(1.0, 1.0, 0.0)
    # points.InsertNextPoint(0.0, 1.0, 0.0)

    # # Create the polygon
    # polygon = vtkPolygon()
    # polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    # polygon.GetPointIds().SetId(0, 0)
    # polygon.GetPointIds().SetId(1, 1)
    # polygon.GetPointIds().SetId(2, 2)
    # polygon.GetPointIds().SetId(3, 3)

    # exterior_pts = kwimage.Boxes.random(1).to_polygons()[0].exterior.data

    # Add the polygon to a list of polygons
    polygons = vtkCellArray()
    points = vtkPoints()

    import vtk
    object_ids = vtk.vtkFloatArray()
    object_ids.SetName('object_ids')

    import shapely
    site_models = list(SiteModel.coerce_multiple('/home/joncrall/temp/debug_smartflow_v2/ingress/sc_out_site_models/'))

    import ubelt as ub
    site_models_dpath = ub.Path('/home/joncrall/data/dvc-repos/smart_expt_dvc/_airflow/preeval21_batch_v197/KR_R002/consolidated_output/site_models/')
    site_models = list(SiteModel.coerce_multiple(site_models_dpath))

    site_geoms = []
    for site_idx, site in enumerate(site_models):
        site_geoms.append(site.geometry)
    union_poly = shapely.ops.unary_union(site_geoms)
    space_bounds = kwimage.MultiPolygon.from_shapely(union_poly).box()

    import numpy as np
    object_ids_ = np.linspace(0, 1, len(site_models))

    min_time = float('inf')
    max_time = -float('inf')

    # https://discourse.vtk.org/t/how-to-triangulate-polygon-with-holes/11497
    # https://examples.vtk.org/site/Python/GeometricObjects/PolyLine/

    for site_idx, site in enumerate(site_models):

        start_time = site.start_date.timestamp()
        end_time = site.end_date.timestamp()

        max_time = max(max_time, end_time)
        min_time = min(min_time, start_time)

        object_id = object_ids_[site_idx]
        poly = kwimage.Polygon.coerce(site.geometry)
        exterior_pts = poly.exterior.data
        twoface = []

        # Add vertexes
        time_points = [start_time, end_time]
        for time_idx in [0, 1]:
            face_idxs = []
            t = time_points[time_idx]
            for idx, xy in enumerate(exterior_pts):
                x, y = xy
                ptr = points.InsertNextPoint(x, y, t)
                object_ids.InsertNextValue(object_id)
                face_idxs.append(ptr)
            twoface.append(face_idxs)

        face1 = twoface[0]
        face2 = twoface[1]

        # Create extrusion faces
        face1_window = list(ub.iter_window(face1, wrap=True))
        face2_window = list(ub.iter_window(face2, wrap=True))
        for w1, w2 in zip(face1_window, face2_window):
            w2_ = w2[::-1]
            # print(w1, w2_)
            flat = w1 + w2_
            polygons.InsertNextCell(len(flat))
            for f in flat:
                polygons.InsertCellPoint(f)

        if 0:
            # Create front / end faces
            from scipy.spatial import Delaunay
            tri = Delaunay(exterior_pts)
            for face_offset in [face1[0], face2[0]]:
                offset_simplies = tri.simplices + face_offset
                for simp in offset_simplies:
                    polygons.InsertNextCell(len(simp))
                    for f in simp:
                        polygons.InsertCellPoint(f)
        else:
            # Hack for the face.
            polygon = poly.to_shapely()

            # Not sure how to de-dup points
            try:
                face_points, face_simplices = triangulate_polygon_interior(polygon)
                for time_idx in [0, 1]:
                    t = time_points[time_idx]
                    ptr_offset = points.GetNumberOfPoints()
                    for x, y in face_points:
                        ptr = points.InsertNextPoint(x, y, t)
                        object_ids.InsertNextValue(object_id)

                    offset_simplicies = face_simplices + ptr_offset
                    for simp in offset_simplicies:
                        polygons.InsertNextCell(len(simp))
                        for f in simp:
                            polygons.InsertCellPoint(f)
            except Exception as ex:
                print(f'ex={ex}')
                raise
                continue

    # Create a PolyData
    polygonPolyData = vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    # polygonPolyData.GetPointData().AddArray(object_ids)
    polygonPolyData.GetPointData().SetScalars(object_ids)

    # LookupTable on mapper to associate scalars with colors

    # Create a mapper and actor
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polygonPolyData)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.5)
    actor.GetProperty().SetColor(colors.GetColor3d('Silver'))

    # Visualize
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName('Polygon')
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d('Salmon'))

    # https://discourse.vtk.org/t/scaling-a-rendering-scene/173/2

    # exterior_pts.max(axis=0)

    cam = renderer.GetActiveCamera()
    transform = vtk.vtkTransform()

    space_unit = max(space_bounds.width, space_bounds.height)

    time_extent = max_time - min_time

    s_x = 1 / space_unit
    s_y = 1 / space_unit
    s_z = 1 / time_extent
    transform.Scale(s_x, s_y, s_z)
    cam.SetModelTransformMatrix(transform.GetMatrix())
    cam.SetParallelProjection(True)

    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()
    renderWindow.Render()
    renderWindowInteractor.Start()

    import vtk
    # Create a vtkXMLPolyDataWriter object
    writer = vtk.vtkXMLPolyDataWriter()
    # Set the input data
    writer.SetInputData(polygonPolyData)
    # Set the file name
    writer.SetFileName("output.vtp")
    # Write the file
    writer.Write()


if __name__ == '__main__':
    main()
