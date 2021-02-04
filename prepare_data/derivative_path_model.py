import pathlib
import pandas as pd


class DerivativePathModel:
    def __init__(self, counter_measure_array):

        path_dict = {}
        # エクセルの名前がうまく作られていないので、応急処置として導入
        # for_data_dict = {}
        counter_measure_dict = {}
        for model_row in counter_measure_array:
            # エクセルの名前がうまく作られていないので、応急処置として導入
            # a001_FM1の場合　{ "a001" : "a001_FM1"}
            # for_data_dict[str(model_row[0].split('_')[0])] = str(model_row[0])

            # a001_FM1の場合　{ "a002" : base_model_path like "a001"}
            print(model_row[0])
            path_dict[str(model_row[0])] = path_dict[str(model_row[1])].joinpath(model_row[0]) \
                if model_row[1] else pathlib.Path(model_row[0])
            counter_measure_dict[str(model_row[0])] = counter_measure_dict[str(model_row[1])] + model_row[3:] \
                if model_row[1] else model_row[3:]
        self.path_dict = path_dict
        # self.for_data_dict = for_data_dict
        self.counter_measure_dict = counter_measure_dict
        last_model = [model_row[0] for model_row in counter_measure_array if model_row[2] == 1][0]

        self.last_model = str(last_model)
    def calc_diff(self, base, derivative, verification=False):
        base_path = self.path_dict[base]
        derivative_path = self.path_dict[derivative]

        # if derivative_path is not based on base_path, raise ValueError
        relative_path = derivative_path.relative_to(base_path)
        deep_level = len(relative_path.parents)
        print(deep_level)
        # for i in range(len(derivative_path.parents)):
        #     if base_path == derivative_path.parents[i]:
        #         deep_level = i + 1
        #
        if not verification and deep_level > 7:
            raise ValueError

        relative_counter_measure = self.counter_measure_dict[derivative] - self.counter_measure_dict[base]
        return relative_counter_measure

    @staticmethod
    def parse_diff_label_for_single_label(diff_label):
        labels = [str(flag) for flag in diff_label]
        diff_label_str = ''.join(labels)
        return diff_label_str

def convert_excel2array(excel_path, column_num=14, sheet_name=None):
    raw_data = pd.read_excel(excel_path, header=1, sheet_name=sheet_name)
    data = raw_data.fillna(0)
    data_array = data.values[1:, 1:4 + column_num]
    return data_array
