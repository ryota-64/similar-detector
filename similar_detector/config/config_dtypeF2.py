import os
import pathlib


class Config(object):

    # model config
    env = 'default'
    # backbone = 'resnet18'
    backbone = 'resnet_face18'
    # backbone = 'resnet50'
    # classify = 'softmax' # 使ってない？
    num_classes = 14
    # metric = 'arc_margin'
    metric = 'linear'
    # easy_margin = False
    easy_margin = True
    use_se = True
    # use_se = False
    # chose loss function
    # loss = 'focal_loss'
    # loss = 'cross_entropy'
    # for multi target?
    loss = "BCEWithLogitsLoss"
    # transfer_train = True
    transfer_train = False
    base_weight_path = '../checkpoints/resnet_face18_celebA_1/resnet18_20.pth'

    # (channel_num, x, y)
    input_shape = (5, 256, 256)

    # optimizer = 'sgd'
    optimizer = 'Adam'

    # for prepare data
    # raw_data_path = 'data/raw_data/20200807'
    # raw_data_path = '../data/raw_data/20201023'
    raw_data_path = '../data/raw_data/20201204'

    # data dir config
    data_sets_dir = '../data/DataSets/'
    # dir_name = 'sample_data' # celebA
    dir_name = 'dtypeF'
    dir_name_for_create_data_sets = 'dtypeF'
    data_is_image = False

    # checkpoints_path = '../checkpoints/resnet_face18_celebA_2/' # arc face
    # test_metric_fc_path = '../checkpoints/resnet_face18_celebA_2/metric_fc_60.pth'
    # test_model_path = '../checkpoints/resnet_face18_celebA_2/resnet_face18_60.pth'
    # checkpoints_path = '../checkpoints/resnet_face18_celebA_3/'  # resnet_face18, linear
    # test_metric_fc_path = '../checkpoints/resnet_face18_celebA_3/metric_fc_60.pth'
    # test_model_path = '../checkpoints/resnet_face18_celebA_3/resnet_face18_60.pth'
    # checkpoints_path = '../checkpoints/resnet_face18_dtypeE/'  # resnet_face18, linear
    # test_metric_fc_path = '../checkpoints/resnet_face18_dtypeE/metric_fc_60.pth'
    # test_model_path = '../checkpoints/resnet_face18_dtypeE/resnet_face18_60.pth'
    checkpoints_path = '../checkpoints/resnet_face18_dtypeF/'  # resnet_face18, linear
    test_metric_fc_path = '../checkpoints/resnet_face18_dtypeF/metric_fc_20.pth'
    test_model_path = '../checkpoints/resnet_face18_dtypeF/resnet_face18_20.pth'

    # other config
    display = True
    # display = False
    finetune = True
    # finetune = False
    use_gpu = True  # use GPU or not
    # use_gpu = False  # use GPU or not
    gpu_id = '0, 1'
    # gpu_id = '0'
    num_workers = 4  # how many workers for loading data
    print_freq = 10  # print info every N batch

    debug_file = 'tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    save_interval = 5

    train_batch_size = 4  # batch size
    test_batch_size = 4

    # # for estimate (openMax)
    # WEIBULL_TAIL_SIZE = 5
    # ALPHA_RAN = 4
    # # Type of distance to be used for calculating distance \
    # # between mean vector and query image \
    # # (eucos, cosine, euclidean)
    # distance_type = 'euclidean'
    # # MVA file path
    # mean_files_path = 'data/mean_files/'
    # # weibull model path
    # weibull_path = 'data/weibull_model/'
    #
    # # Path to directory where distances of training data from Mean Activation Vector is saved
    # distance_path = 'data/mean_distance_files/'
    # feature_path = 'data/train_features/'
    # euc_scale = 5e-3
    # SCORE_THRESHOLD = 100
    # SCORE_NORMALIZED = 0.7

    def __init__(self, root_path=pathlib.Path(__file__).parents[1], for_prepare_data_creation=False):
        # self.device = self.device if torch.cuda.is_available() else 'cpu'
        self.data_sets_dir = os.path.join(root_path, self.data_sets_dir)
        self.raw_data_path = os.path.join(root_path, self.raw_data_path)

        # preapare_data.pyの時とで場合分け
        self.dir_name = self.dir_name_for_create_data_sets if for_prepare_data_creation else self.dir_name
        self.criteria_list = os.path.join(self.data_sets_dir, self.dir_name, 'criteria_list.txt')
        self.train_root = os.path.join(self.data_sets_dir, self.dir_name, 'train/models')
        self.train_list = os.path.join(self.data_sets_dir, self.dir_name, 'train/train_labels.json')
        self.val_list = os.path.join(self.data_sets_dir, self.dir_name, 'train/val_labels.json')
        self.test_root = os.path.join(self.data_sets_dir, self.dir_name, 'train/models')
        self.test_list = os.path.join(self.data_sets_dir, self.dir_name, 'train/val_labels.json')
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
