

from logging import getLogger
import pathlib

import numpy as np

from config import Config
from data.prepare_data import PlateData, DynainData

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
            conter_itr = conter_dir.glob('{}.csv'.format(dynain_path.stem[:-7]))
            conter_paths.append(_take_one_or_ret_false(conter_itr))
    return conter_paths


# 一個だけ存在すればそれを返す、それ以外ならfalse
# todo どれのblankなのかをうまく判定できるように
def get_blank_csv(dynain_path):
    blank_nodes = dynain_path.parents[1].joinpath('blank/NodeID').glob('{}*_BLANK_*.csv'.format(dynain_path.stem[:1]))

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
                plate_data.set_conter(conter.parents[0].stem, conter)

            output = plate_data.output()
            # extract data and save it
            # todo 一部をtest用のデータセットに保存する
            np.save('data/DataSets/dtypeA/train/models/{}_plate_data.npy'.format(dynain_path.stem[:-7]), output)
        except KeyError as e:
            print(dynain_path, e)


if __name__ == '__main__':
    main()
