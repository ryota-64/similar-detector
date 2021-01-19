from logging import getLogger

from __future__ import print_function
import os
import time

import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import DataParallel
# from sklearn.metrics import multilabel_confusion_matrix

from .models import *
from .config import Config
from .datasets import dataset


logger = getLogger(__name__)




def main():
    opt = Config()
    if torch.cuda.is_available() and opt.use_gpu:  # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    print('device: {}'.format(device))

    # model setup

    if opt.backbone == 'resnet_face18':
        model = resnet_face18(opt.input_shape[0], use_se=opt.use_se)
    elif opt.backbone == 'resnet18':
        model = resnet18(opt.input_shape[0], pretrained=False)
    elif opt.backbone == 'resnet34':
        model = resnet34(opt.input_shape[0])
    elif opt.backbone == 'resnet50':
        model = resnet50(opt.input_shape[0])
    else:
        raise TypeError('not match model type')
    model.to(device)
    if device == 'cuda':
        model = DataParallel(model)

    # load weight
    # if device == 'cuda':
    #     model.load_state_dict(torch.load(opt.test_model_path))
    # else:
    #     model.load_state_dict(torch.load(opt.test_model_path, map_location={'cuda:0': 'cpu'}))

    model.eval()

    # metric_fc area
    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    if device == 'cuda':
        metric_fc = DataParallel(metric_fc)
    metric_fc.to(device)

    # # load weight
    # if device == 'cuda':
    #     metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path))
    # else:
    #     metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path, map_location={'cuda:0': 'cpu'}))
    metric_fc.eval()

    from torchviz import make_dot
    x = torch.randn(1,6,256,256)
    y = model(x)
    dot= make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('graph_image')

if __name__ == '__main__':
    main()
