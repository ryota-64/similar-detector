from logging import getLogger


import pymesh

from prepare_data import DynainData, PlateData
logger = getLogger(__name__)

if __name__ == '__main__':
    blank_node_path = 'data/raw_data/20201023/blank/NodeID/a2_BLANK_NodeID.csv'
    # conter_paths = get_conter_csv(dynain_path)
    # print(dynain_path, blank_node_path, conter_paths)
    dynain_path = 'data/raw_data/20201023/dynain/a2_FM1_dynain'
    dynain_data = DynainData(dynain_path)
    plate_data = PlateData(blank_node_path)
    plate_data.set_dynain_data(dynain_data)

    vertices, faces = dynain_data._data_for_pymesh()
    mesh = pymesh.form_mesh(vertices, faces)

    print(mesh.num_vertices, mesh.num_faces, mesh.num_voxels)
    triangled_mesh = pymesh.quad_to_tri(mesh)
    triangled_mesh.add_attribute("vertex_gaussian_curvature")
    print(triangled_mesh.get_attribute("vertex_gaussian_curvature").shape)
