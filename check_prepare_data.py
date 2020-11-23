from logging import getLogger

import numpy as np

from prepare_data.plate_data import DynainData, PlateData

logger = getLogger(__name__)

if __name__ == '__main__':
    blank_node_path = 'data/raw_data/20201023/blank/NodeID/a2_BLANK_NodeID.csv'
    # conter_paths = get_conter_csv(dynain_path)
    # print(dynain_path, blank_node_path, conter_paths)
    dynain_path = 'data/raw_data/20201023/dynain/a2_FM1_dynain'
    dynain_data = DynainData(dynain_path)
    plate_data = PlateData(blank_node_path)
    plate_data.set_dynain_data(dynain_data)
    plate_data.set_dynain_data_old(dynain_data)
    # for conter in conter_paths:
    #     print(
    #         conter.relative_to(opt.raw_data_path).parents[len(conter.relative_to(opt.raw_data_path).parents) - 3].name)
    #     plate_data.set_conter(
    #         conter.relative_to(opt.raw_data_path).parents[len(conter.relative_to(opt.raw_data_path).parents) - 3].name,
    #         conter)
    plate_data.plot_normal_vecotr(save_name='result/vector_sample.png')
    output = plate_data.output(output_size=(256, 256))
    np.save('result/out.npy', output)
    plate_label = plate_data.output_labels()
    print('lo')
    print(output.shape)

    normal_vec_fig = plate_data.get_normal_vector_fig(figsize=(32, 32))

    from matplotlib import pyplot as plt
    plt.imshow(normal_vec_fig)
    plt.show()
