from logging import getLogger
import pathlib
import numpy as np

from prepare_data.plate_data import DynainData, PlateData

logger = getLogger(__name__)

if __name__ == '__main__':
    # blank_node_path = 'data/raw_data/20201218/A1_150_Bhinge/a001_FM1/a001_FM1_blank.csv'
    # conter_paths = get_conter_csv(dynain_path)
    # print(dynain_path, blank_node_path, conter_paths)
    # dynain_path = 'data/raw_data/20201218/A1_150_Bhinge/a001_FM1/a001_FM1_dynain'
    model_dir = pathlib.Path('/prepare/similar-detector/data/raw_data/20201222/D1_Ahinge/D40_FM1')

    dynain_path = model_dir.joinpath('{}_dynain'.format(model_dir.stem))
    conter_paths = [model_dir.joinpath('{}_{}.csv'.format(model_dir.stem, i + 1)) for i in range(4)]
    blank_node_path = model_dir.joinpath('{}_blank.csv'.format(model_dir.stem))

    # model_dir = pathlib.Path('data/raw_data/20201204/1_HINGE_RF_170801/d10/')
    dynain_data = DynainData(dynain_path)
    plate_data = PlateData(blank_node_path)
    plate_data.set_dynain_data(dynain_data)
    for conter in conter_paths:
        plate_data.set_conter(conter.name, conter)

    output = plate_data.output(output_size=(256, 256))
    np.save('result/out3.npy', output)
