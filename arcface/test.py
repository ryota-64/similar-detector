# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import time

import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import DataParallel
from sklearn.metrics import multilabel_confusion_matrix

from .models import *
from .config import Config
from .datasets import dataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return 'avg : {}\tnum : {}'.format(self.avg, self.count)


def accuracy(preds, labels):
    accs = []
    for i in range(len(preds)):
        accs.append(preds[i] == labels[i])
    print(accs.count(True))
    return accs.count(True)


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


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
    if device == 'cuda':
        model.load_state_dict(torch.load(opt.test_model_path))
    else:
        model.load_state_dict(torch.load(opt.test_model_path, map_location={'cuda:0': 'cpu'}))

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

    # load weight
    if device == 'cuda':
        metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path))
    else:
        metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path, map_location={'cuda:0': 'cpu'}))
    metric_fc.eval()

    # data loader todo train_root →　test_rootに (yet to move data to test_root)
    test_dataset = dataset.DataSet(opt.train_root, opt.val_list, phase='test', input_shape=opt.input_shape,
                                   data_is_image=opt.data_is_image)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=opt.test_batch_size,
                                              shuffle=False,
                                              num_workers=opt.num_workers)

    accs = AverageMeter()
    # predicted labels
    pred_list = []
    # answer labels
    labels = []
    acc_individual = {}
    data_path_list = []
    for ii, test_batch in enumerate(tqdm(test_loader)):
        data_input, label, data_path = test_batch
        # data_path_list.append(data_path)
        data_input = data_input.to(device)
        label = label.to(device).long()
        feature = model(data_input)
        if opt.metric == 'linear':
            output_labels = metric_fc(feature)
        else:
            output_labels = metric_fc(feature, label)
        output = output_labels.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc = np.mean((output == label).astype(int))
        preds = output
        # preds = preds.to(torch.float32)
        # preds = preds.to('cpu').detach().numpy().copy()
        # label = label.to('cpu').detach().numpy().copy()

        # print(len(label[0]))
        # print(len(preds[0]))

        labels.extend(label)
        pred_list.extend(preds)
        data_path_list.extend(data_path)
    # 正解ラベルと照合

    confusion_matrix = multilabel_confusion_matrix(labels, pred_list)
    print(confusion_matrix)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from itertools import chain
    print('total accuracy: ', accuracy_score(list(labels), list(pred_list)))
    print('precision score: ', precision_score(labels, pred_list, average='micro'))
    print('recall score: ', recall_score(labels, pred_list, average='micro'))
    print('f1 score: ', f1_score(labels, pred_list, average='micro'))
    #
    # for label, pred, data_path in zip (labels, pred_list,data_path_list ):
    #     acc = 0
    #     for l, p in zip (label, pred):
    #         if l == p:
    #             acc +=1
    #
    #     print(acc /14, data_path[0][-25:])
    print(len(labels))
    print(pred_list[0])

    out_data = np.array(
        np.concatenate([np.array(data_path_list)[:,np.newaxis], np.concatenate([np.array(labels)[:,np.newaxis], np.array(pred_list)[:,np.newaxis]], axis=1)], axis=1),
        dtype=object)

    np.save('./result/{}_result.npy'.format(opt.dir_name), out_data)


    a = np.array(labels)
    print(a.shape)
    b = np.sum(a, axis=0)
    print(b.shape)
    print(b)

    # acc = accuracy(pred_list, labels)
    # acc_individual = accuracy_indivisual(pred_list, labels)
    # print(acc / len(pred_list))
    #
    # for key, value in acc_individual.items():
    #     print('key: {}  {}/{}  {}%\n'.format(key, value.count(True), len(value), value.count(True) / len(value) * 100))
    #
    # print('only_train_label\n')
    # for key, value in acc_individual.items():
    #     if key in centroid_map:
    #         print(
    #             'key: {}  {}/{}  {}%'.format(key, value.count(True), len(value), value.count(True) / len(value) * 100))
    #         accs.update(value.count(True) / len(value), len(value))
    # print(accs)
