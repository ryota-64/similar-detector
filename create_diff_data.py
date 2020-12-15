from prepare_data.derivative_path_model import DerivativePathModel, convert_excel2array


if __name__ == '__main__':
    excel_path = './data/raw_data/20201204/対策対応表(model-base).xlsx'
    counter_measures_array = convert_excel2array(excel_path)
    derivative_paths = DerivativePathModel(counter_measures_array)

    # derivative_paths.calc_diff("Hinge_g39", "Hinge_g37")
    print(derivative_paths.calc_diff("Hinge_g39", "Hinge_g46"))
    # print(derivative_paths.counter_measure_dict)
