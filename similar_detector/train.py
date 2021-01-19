from __future__ import print_function

import argparse
import os
import pathlib
import time

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR

from .config import Config
from .datasets import DataSet
from .models import *
from .utils import Visualizer


def save_model(model, save_path, name, iter_cnt):
    save_name = pathlib.Path(save_path).joinpath(name + '_' + str(iter_cnt) + '.pth')
    save_name.parents[0].mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_name)
    return save_name


def train(args):
    opt = Config()
    # opt.num_classes = len(get_train_labels(opt.train_root, opt.criteria_list))

    if opt.display:
        # import subprocess
        # subprocess.run(["python", "-m", "visdom.server"])
        visualizer = Visualizer()
    else:
        visualizer = None

    if torch.cuda.is_available() and opt.use_gpu:  # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    print('device: {}'.format(device))

    train_dataset = DataSet(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape,
                            data_is_image=opt.data_is_image)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=opt.train_batch_size,
                                              shuffle=True,
                                              num_workers=opt.num_workers,
                                              drop_last=True)

    val_dataset = DataSet(opt.train_root, opt.val_list, phase='val', input_shape=opt.input_shape,
                          data_is_image=opt.data_is_image)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.test_batch_size,
                                             shuffle=True,
                                             num_workers=opt.num_workers)

    test_dataset = DataSet(opt.train_root, opt.test_list, phase='test', input_shape=opt.input_shape,
                          data_is_image=opt.data_is_image)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=opt.test_batch_size,
                                             num_workers=opt.num_workers)

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    elif opt.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet_face18':
        model = resnet_face18(opt.input_shape[0], use_se=opt.use_se)
    elif opt.backbone == 'resnet18':
        from torchvision import models
        model = models.resnet18(pretrained=True)
        # model = resnet18(opt.input_shape[0], pretrained=True)
    elif opt.backbone == 'resnet34':
        model = resnet34(opt.input_shape[0])
    elif opt.backbone == 'resnet50':
        model = resnet50(opt.input_shape[0])
    else:
        raise TypeError('not match model type')
    model.to(device)
    if device == 'cuda':
        model = DataParallel(model)

    # 転移学習(途中から)
    if opt.transfer_train:
        if device == 'cuda':
            model.load_state_dict(torch.load(opt.base_weight_path))
        else:
            model.load_state_dict(torch.load(opt.test_model_path, map_location={'cuda:0': 'cpu'}))

    if args.train_second:
        opt.metric = 'liner'
        if device == 'cuda':
            model.load_state_dict(torch.load(opt.test_model_path))
        else:
            model.load_state_dict(torch.load(opt.test_model_path, map_location={'cuda:0': 'cpu'}))
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    print(model)
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


    # load weight
    if opt.transfer_train:
        if device == 'cuda':
            metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path))
        else:
            metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path, map_location={'cuda:0': 'cpu'}))
    metric_fc.train()

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch):
        model.train()
        train_acc = []
        train_loss = []
        train_spend = []
        for ii, train_batch in enumerate(tqdm(trainloader)):
            data_input, label = train_batch
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            if args.train_second or opt.metric == 'linear':
                output = metric_fc(feature)
            else:
                output = metric_fc(feature, label)
            loss = criterion(output, label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            # output = np.argmax(output, axis=1)
            output = torch.sigmoid(output).data > 0.5
            output = output.to(torch.float32)
            output = output.data.cpu().numpy()
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            spend_time = (time.time() - start)
            train_acc.extend([acc] * len(label))
            train_loss.extend([loss.item()] * len(label))
            train_spend.extend([spend_time])
            start = time.time()
        time_str = time.asctime(time.localtime(time.time()))
        print('{} train epoch {}  {} seconds/epoch loss {} acc {}'.format(
            time_str, i, np.sum(train_spend), np.mean(train_loss), np.mean(train_acc)))
        if opt.display:
            visualizer.display_current_results(i, np.mean(train_loss), name='train_loss')
            visualizer.display_current_results(i, np.mean(train_acc), name='train_acc')

        scheduler.step()
        if args.train_second:
            save_model(metric_fc, opt.checkpoints_path, 'fc', i)
        else:
            if i % opt.save_interval == 0 or i == opt.max_epoch:
                save_model(model, opt.checkpoints_path, opt.backbone, i)
                save_model(metric_fc, opt.checkpoints_path, 'metric_fc', i)

        model.eval()
        eval_acc = []
        eval_loss = []
        eval_speed = []

        for ii, val_batch in enumerate(val_loader):
            data_input, label, data_path = val_batch
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            if args.train_second or opt.metric == 'linear':
                output = metric_fc(feature)
            else:
                output = metric_fc(feature, label)
            loss = criterion(output, label.float())

            output = output.data.cpu().numpy()
            # output = np.argmax(output, axis=1)
            output = output > 0.5
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            speed = opt.print_freq / (time.time() - start)
            eval_acc.extend([acc] * len(label))
            eval_loss.extend([loss.item()] * len(label))
            eval_speed.extend([speed] * len(label))
        time_str = time.asctime(time.localtime(time.time()))
        # ↓　変なのはいる？acc
        # print('{} val epoch {}  loss {} acc {}'.format(time_str, i,
        #                                                np.mean(eval_speed), np.mean(eval_loss), np.mean(eval_acc)))
        if opt.display:
            visualizer.display_current_results(i, np.mean(eval_loss), name='val_loss')
            visualizer.display_current_results(i, np.mean(eval_acc), name='val_acc')

        test_acc = []
        test_loss = []
        test_speed = []

        for ii, test_batch in enumerate(test_loader):
            data_input, label, data_path = test_batch
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            if args.train_second or opt.metric == 'linear':
                output = metric_fc(feature)
            else:
                output = metric_fc(feature, label)
            loss = criterion(output, label.float())

            output = output.data.cpu().numpy()
            # output = np.argmax(output, axis=1)
            output = output > 0.5
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            speed = opt.print_freq / (time.time() - start)
            test_acc.extend([acc] * len(label))
            test_loss.extend([loss.item()] * len(label))
            test_speed.extend([speed] * len(label))
        time_str = time.asctime(time.localtime(time.time()))
        if opt.display:
            visualizer.display_current_results(i, np.mean(test_loss), name='test_loss')
            visualizer.display_current_results(i, np.mean(test_acc), name='test_acc')


def main():
    parser = argparse.ArgumentParser()

    # Optional arguments.
    parser.add_argument(
        '--train_second',
        action='store_true',
        help='if you want to train with freezing model except full connected layer , be on the flag'
    )

    args = parser.parse_args()
    train(args)
