from prepare_data.derivative_path_model import DerivativePathModel, convert_excel2array
import pathlib
import os

import openpyxl
import numpy as np

from similar_detector.config.config import Config


def load_and_calc_diff(base_path, derivative_path):
    base = np.load(base_path)
    derivative = np.load(derivative_path)

    ret = derivative - base

    return np.concatenate([ret, base[:, :, 0:1]], axis=2)


if __name__ == '__main__':
    opt = Config()
    if opt.ML_model_type == 'diff':
        raw_data_path_list = opt.raw_data_path_list
        print(raw_data_path_list)
        for raw_data_path in raw_data_path_list:
            raw_data_path = pathlib.Path(raw_data_path)
            excel_paths = raw_data_path.glob('*.xlsx')
            for excel_path in excel_paths:
                print(excel_path)
                # excel_path = './data/raw_data/20201204/対策対応表(model-base).xlsx'
                bk = openpyxl.load_workbook(excel_path)
                sheet_names = bk.sheetnames
                for sheet_name in sheet_names:

                    if sheet_name == 'sample':
                        continue
                    print('aaaaaaa')
                    counter_measures_array = convert_excel2array(excel_path, sheet_name=sheet_name)
                    derivative_paths = DerivativePathModel(counter_measures_array)
                    # print(derivative_paths.path_dict.keys())
                    # print(len(derivative_paths.for_data_dict.values()))
                    # print(derivative_paths.for_data_dict.values())
                    detected_paths = []
                    # print(len(list(derivative_paths.path_dict.keys())))
                    for path in derivative_paths.for_data_dict.values():

                        dataset_path = list(raw_data_path.glob('**/{}'.format(path)))
                        # if len(list(dataset_path))!=7:
                        if len(dataset_path) == 0:
                            print('csvのファイルがないパス: {}'.format(path))
                        #     print(list(dataset_path))
                        for _path in dataset_path:
                            if _path.is_dir():
                                detected_paths.append(_path)
                            else:
                                print(_path)
                    print(len(detected_paths))

                    for base_path in detected_paths:
                        for deri_path in detected_paths:
                            if base_path == deri_path:
                                continue
                            try:
                                try:
                                    diff_label = derivative_paths.calc_diff(base_path.name.split('_')[0],
                                                                            deri_path.name.split('_')[0])
                                except ValueError as e:
                                    # print(e)
                                    continue
                                    # todo make dataset path
                                base_file_name = '{}_{}_plate_data.npy'.format(sheet_name, base_path.stem)
                                base_path_from_dateset = pathlib.Path(
                                    os.path.join(opt.origin_data_sets, base_file_name))
                                deriva_file_name = '{}_{}_plate_data.npy'.format(sheet_name, deri_path.stem)
                                deri_path_from_dateset = pathlib.Path(
                                    os.path.join(opt.origin_data_sets, deriva_file_name))
                                diff_data = load_and_calc_diff(base_path_from_dateset, deri_path_from_dateset)

                            except FileNotFoundError as e:
                                # print(e)
                                continue
                            out_put_path = pathlib.Path(opt.data_sets_dir).joinpath(opt.dir_name).joinpath(
                                'train/models').joinpath('{}_{}_diff_data.npy'.format(deri_path.name, base_path.name))
                            out_put_path.parents[0].mkdir(exist_ok=True, parents=True)
                            # print('save to {}'.format(out_put_path))
                            np.save(out_put_path, diff_data)
                    # todo データのパスのリストを作成、それをfor ループで回して,大丈夫なやつだけ、調べる
                    # todo そのあと保存

                    # derivative_paths.calc_diff("Hinge_g39", "Hinge_g37")
                    # print(derivative_paths.calc_diff("Hinge_g39", "Hinge_g46"))
                    # print(derivative_paths.counter_measure_dict)
