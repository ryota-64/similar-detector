import json
from logging import getLogger
import os
import pathlib

import numpy as np

from similar_detector.config import Config
from prepare_data import PlateData, DynainData

logger = getLogger(__name__)

opt = Config()


# 必要なデータがあるかのcheck
def check_data(dynain_path, verbose=False):
    """
    :param dynain_path: pathlib.Path object
    :param verbose: if True, print missed dynains
    :return: bool
    """
    blank_node = get_blank_csv(dynain_path)

    conter_paths = get_conter_csv(dynain_path)
    if not all([blank_node, *conter_paths]) and verbose:
        print(dynain_path)
        print(blank_node, conter_paths)
    return all([blank_node, *conter_paths])


def get_conter_csv(dynain_path):
    conter_paths = []
    # todo 必要なconterをconfigで指定するようにする？
    for conter_dir in dynain_path.parents[1].joinpath('conters/').iterdir():
        if conter_dir.stem[:1] != '.':
            conter_itr = conter_dir.glob('**/{}*.csv'.format(dynain_path.stem.split('_')[0]))
            conter_paths.append(_take_one_or_ret_false(conter_itr))
    return conter_paths


# 一個だけ存在すればそれを返す、それ以外ならfalse
# todo どれのblankなのかをうまく判定できるように
def get_blank_csv(dynain_path):
    blank_nodes = dynain_path.parents[1].joinpath('blank/NodeID').glob(
        '{}*_BLANK_*.csv'.format(dynain_path.stem.split('_')[0]))

    return _take_one_or_ret_false(blank_nodes)


# iteratorのlengthが１ならその要素を,それ以外ならfalseを返す
def _take_one_or_ret_false(iterator):
    try:
        one = iterator.__next__()
    except StopIteration:
        return False

    # 2個目以上あれば、false
    if sum(1 for _ in iterator) == 0:
        return one
    else:
        return False


def main():
    # check hire
    raw_data_path = pathlib.Path(opt.raw_data_path)
    dynains = [dynain_path for dynain_path in raw_data_path.joinpath('dynain').iterdir() if check_data(dynain_path)]

    print(dynains)
    label_data = {}
    # get list of data
    for dynain_path in dynains:
        try:
            blank_node_path = get_blank_csv(dynain_path)
            conter_paths = get_conter_csv(dynain_path)
            print(dynain_path, blank_node_path, conter_paths)
            dynain_data = DynainData(dynain_path)
            plate_data = PlateData(blank_node_path)
            plate_data.set_dynain_data(dynain_data)
            for conter in conter_paths:
                print(conter.relative_to(opt.raw_data_path).parents[
                          len(conter.relative_to(opt.raw_data_path).parents) - 3].name)
                plate_data.set_conter(conter.relative_to(opt.raw_data_path).parents[
                                          len(conter.relative_to(opt.raw_data_path).parents) - 3].name, conter)

            output = plate_data.output(output_size=(256, 256))

            plate_label = plate_data.output_labels()
            # extract data and save it
            # todo 一部をtest用のデータセットに保存する
            file_name = '{}_plate_data.npy'.format(dynain_path.stem[:-7])
            output_path = pathlib.Path(os.path.join(opt.data_sets_dir, opt.dir_name, 'train/models', file_name))
            if not output_path.exists() and not output_path.parents[0].is_dir():
                output_path.parents[0].mkdir(parents=True)
            np.save(output_path, output)
            label_data[file_name] = plate_label

        except KeyError as e:
            print(dynain_path, e)

    pathlib.Path(opt.train_list).parents[0].mkdir(parents=True, exist_ok=True)
    with open(opt.train_list, mode='w')as f:
        json.dump({'data': label_data}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
