
import os
import pathlib
import yaml


class Config(object):

    def __init__(self, root_path=pathlib.Path(__file__).parents[1]):
        default_config_path = pathlib.Path(__file__).parent.joinpath('./default_config.yml')
        with open(default_config_path, mode='r')as f:
            config_json = yaml.load(f)
        if config_json['custom_config']:
            with open(os.path.join(root_path, config_json['custom_config']), mode='r')as f1:
                custom_config = yaml.load(f1)
            for key, value in custom_config.items():
                config_json[key] = value

        self.input_shape = (config_json.pop('input_shape_channel'),
                            config_json.pop('input_shape_x'), config_json.pop('input_shape_y'))
        # print(config_json)
        for key, value in config_json.items():
            setattr(self, key, value)

        self.lr = 1e-1  # initial learning rate
        self.lr_step = 10
        self.lr_decay = 0.95  # when val_loss increase, lr : lr*lr_decay
        self.weight_decay = 5e-4

        self.origin_data_sets = os.path.join(root_path, self.origin_data_sets) # for diff_type
        # self.device = self.device if torch.cuda.is_available() else 'cpu'
        self.data_sets_dir = os.path.join(root_path, self.data_sets_dir)
        self.raw_data_path_list = [os.path.join(root_path, raw_data_path) for raw_data_path in self.raw_data_path_list]

        # preapare_data.pyの時とで場合分け
        # self.dir_name = self.dir_name_for_create_data_sets if for_prepare_data_creation else self.dir_name
        self.criteria_list = os.path.join(self.data_sets_dir, self.dir_name, 'criteria_list.txt')
        self.train_root = os.path.join(self.data_sets_dir, self.dir_name, 'train/models')
        self.train_list = os.path.join(self.data_sets_dir, self.dir_name, 'train/train_labels.json')
        self.val_list = os.path.join(self.data_sets_dir, self.dir_name, 'train/val_labels.json')
        self.test_root = os.path.join(self.data_sets_dir, self.dir_name, 'train/models')
        self.test_list = os.path.join(self.data_sets_dir, self.dir_name, 'test/test_labels.json')
        self.test_model_path = os.path.join(root_path, self.test_model_path)
        self.test_metric_fc_path = os.path.join(root_path, self.test_metric_fc_path)
        self.debug_file = os.path.join(root_path, self.debug_file)
        self.checkpoints_path = os.path.join(root_path, self.checkpoints_path)
        self.base_weight_path = os.path.join(root_path, self.base_weight_path)
        # temp comment out
        # self.num_classes = len([dir_name for dir_name in os.listdir(self.train_root)
        #                         if pathlib.Path(self.train_root).joinpath(dir_name).is_dir()])
        # self.num_classes = len(list(pathlib.Path(self.raw_data_path).joinpath('conters').iterdir()))
        # self.mean_files_path = os.path.join(root_path, self.mean_files_path)
        # self.distance_path = os.path.join(root_path, self.distance_path)
        # self.feature_path = os.path.join(root_path, self.feature_path)
