import pathlib
import pandas as pd


class DerivativePathModel:
    def __init__(self, counter_measure_array):

        path_dict = {}
        counter_measure_dict = {}
        for model_row in counter_measure_array:
            path_dict[str(model_row[0])] = path_dict[str(model_row[1])].joinpath(model_row[0]) \
                if model_row[1] else pathlib.Path(model_row[0])
            counter_measure_dict[str(model_row[0])] = counter_measure_dict[str(model_row[1])] + model_row[2:] \
                if model_row[1] else model_row[2:]
        self.path_dict = path_dict
        self.counter_measure_dict = counter_measure_dict

    def calc_diff(self, base, derivative):
        base_path = self.path_dict[base]
        derivative_path = self.path_dict[derivative]

        # if derivative_path is not based on base_path, raise ValueError
        relative_path = derivative_path.relative_to(base_path)

        relative_counter_measure = self.counter_measure_dict[derivative] - self.counter_measure_dict[base]
        return relative_counter_measure


def convert_excel2array(excel_path, column_num=14):
    raw_data = pd.read_excel(excel_path, header=1, sheet_name='sheet2')
    data = raw_data.fillna(0)
    data_array = data.values[1:, :2 + column_num]
    return data_array
