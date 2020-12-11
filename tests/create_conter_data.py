from logging import getLogger
import pathlib
import numpy as np

from prepare_data.plate_data import DynainData, PlateData

logger = getLogger(__name__)

if __name__ == '__main__':

    model_path = pathlib.Path('data/raw_data/20201204/1_HINGE_RF_170801/d10/')
    blank_node_path = model_path.joinpath('{}_blank.csv'.format(model_path.stem))
    # conter_paths = get_conter_csv(dynain_path)
    # print(dynain_path, blank_node_path, conter_paths)
    dynain_path = model_path.joinpath('{}_dynain'.format(model_path.stem))

    conter_paths = [model_path.joinpath('{}_{}.csv'.format(model_path.stem, i + 1)) for i in range(4)]
    dynain_data = DynainData(dynain_path)
    plate_data = PlateData(blank_node_path)
    plate_data.set_dynain_data(dynain_data)
    for conter in conter_paths:
        plate_data.set_conter(conter.name, conter)

    conter_name = conter_paths[1].name
    PlateData.plot_plate_conter(plate_data.get_plate_conter(conter_name), save_name='./result/{}.png'.format(conter_name))


    #
    # plate_data.plot_normal_vector(save_name='result/vector_sample.png')
    # output = plate_data.output(output_size=(256, 256))
    # np.save('result/out3.npy', output)
    # # plate_label = plate_data.output_labels()
    # print('lo')
    # print(output.shape)
    #
    # normal_vec_fig = plate_data.get_normal_vector_fig(figsize=(32, 32))
    #
    # from matplotlib import pyplot as plt
    # plt.imshow(normal_vec_fig)
    # plt.show()
