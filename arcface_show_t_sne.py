import time
import os

import numpy as np

import torch
from torch.utils import data
from torch.nn import DataParallel

from arcface.datasets import DataSet
from arcface.config import Config
from arcface.models import *

from sklearn.manifold import TSNE
from sklearn import preprocessing
import umap
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as offline


# import plotly.graph_objects as go

# offline.init_notebook_mode()
#
# from metric_learning.src.try_network import MetricNN
# from metric_learning.src.try_network import Tripletnet


def main():
    opt = Config()
    if opt.backbone == 'resnet_face18':
        model = resnet_face18(opt.input_shape[0], use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34(opt.input_shape[0])
    elif opt.backbone == 'resnet50':
        model = resnet50(opt.input_shape[0])
    else:
        raise TypeError('not match model type')
    model.to(device)
    # load_model(model, opt.test_model_path)
    # if torch.cuda.is_available() and opt.use_gpu == 'cuda':
    print(opt.use_gpu)
    print(device)
    if opt.use_gpu:
        model = DataParallel(model)
        # model.load_state_dict(torch.load(opt.test_model_path, map_location={'cuda:0': 'cpu'}))
        model.load_state_dict(torch.load(opt.test_model_path))
    else:
        model.load_state_dict(torch.load(opt.test_model_path, map_location={'cpu': 'cpu'}))
    # model.to(torch.device(device))
    model.eval()
    global args

    train_dataset = DataSet(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    # centroid_map = create_centroid(model, trainloader)

    val_dataset = DataSet(opt.train_root, opt.val_list, phase='test', input_shape=opt.input_shape,
                          data_is_image=opt.data_is_image)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.test_batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)

    test_dataset = DataSet(opt.test_root, opt.test_list, phase='test', input_shape=opt.input_shape)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=4,
                                  # batch_size=opt.test_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    latent_vecs_list = np.empty((0, 512))
    label_list = np.empty((0))
    for i, test_batch in enumerate(trainloader):
        data_input, label = test_batch
        data_input = data_input.to(device)
        # label = label.to(device).long()
        latent_vecs = model(data_input)
        latent_vecs_list = np.concatenate([latent_vecs_list, latent_vecs.cpu().detach().numpy()])
        label_list = np.concatenate([label_list, label])
        target = label
        # plot3d_tsne(latent_vecs, target, )
        # show_umap(latent_vecs, target)

    plotly_t_sne(latent_vecs_list, label_list, data_type='train', dir_type=opt.dir_name.replace('/', '-'))
    t_sne(latent_vecs_list, label_list, data_type='train', dir_type=opt.dir_name.replace('/', '-'))
    # t_sne(latent_vecs, target)

    latent_vecs_list = np.empty((0, 512))
    label_list = []
    for i, test_batch in enumerate(val_loader):
        data_input, label, data_path = test_batch
        data_input = data_input.to(device)
        # label = label.to(device).long()
        latent_vecs = model(data_input)
        latent_vecs_list = np.concatenate([latent_vecs_list, latent_vecs.cpu().detach().numpy()])
        target = label
        label_list.append(label)
        # plot3d_tsne(latent_vecs, target, )
        # show_umap(latent_vecs, target)
        # t_sne(latent_vecs, target)
    label_array = np.concatenate(label_list)

    t_sne(latent_vecs_list, label_array, data_type='val', dir_type=opt.dir_name.replace('/', '-'))
    plotly_t_sne(latent_vecs_list, label_array, data_type='val', dir_type=opt.dir_name.replace('/', '-'))

    latent_vecs_list = np.empty((0, 512))
    label_list = []
    for i, test_batch in enumerate(test_loader):
        data_input, label, data_path = test_batch
        data_input = data_input.to(device)
        # label = label.to(device).long()
        latent_vecs = model(data_input)
        latent_vecs_list = np.concatenate([latent_vecs_list, latent_vecs.cpu().detach().numpy()])
        target = label
        label_list.append(label)
        # plot3d_tsne(latent_vecs, target, )
        # show_umap(latent_vecs, target)
        # t_sne(latent_vecs, target)
    label_array = np.concatenate(label_list)

    plotly_t_sne(latent_vecs_list, label_array, data_type='test', dir_type=opt.dir_name.replace('/', '-'))
    t_sne(latent_vecs_list, label_array, data_type='test', dir_type=opt.dir_name.replace('/', '-'))


#
#     for x, y in test_loader:
#         print(x.shape)
#         latent_vecs = model(x)
#         print(latent_vecs.shape, y.shape)
#         target = y
#         plot3d_tsne(latent_vecs, target,)
#         show_umap(latent_vecs, target)
#         t_sne(latent_vecs, target)
# #
#
# def show_t_SNE_umap(test_loader_path, model, tripletnet):
#     if torch.cuda.is_available():  # GPUが利用可能か確認
#         device = 'cuda'
#     else:
#         device = 'cpu'
#     test_image_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(
#             root=test_loader_path,
#             transform=transforms.Compose([
#                     transforms.Resize([256, 256]),
#                     transforms.Grayscale(),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.1307,), (0.3081,)),
#
#                 ]),
#         ),
#         batch_size=1000
#     )
#
#     data_type = pathlib.Path(test_loader_path).stem
#     model.to(device)
#     tripletnet.to(device)
#     metric_model = tripletnet.embeddingnet
#
#     for x, y in test_image_loader:
#
#         latent_vecs = metric_model.forward(x)
#         target = y
#         plot3d_tsne(latent_vecs, target, data_type)
#         show_umap(latent_vecs, target, data_type)
#         t_sne(latent_vecs, target, data_type)
#         # t_sne(x, y)


def plot3d_tsne(latent_vecs, target, data_type='test'):
    latent_vecs = latent_vecs.to("cpu")
    latent_vecs = latent_vecs.detach().numpy()
    start_time = time.time()
    tsne = TSNE(n_components=3, random_state=0).fit_transform(latent_vecs)

    # 3Dの散布図が作れるScatter3dを使います．
    trace1 = go.Scatter3d(
        x=tsne[:, 0],  # それぞれの次元をx, y, zにセットするだけです．
        y=tsne[:, 1],
        z=tsne[:, 2],
        mode='markers',
        marker=dict(
            sizemode='diameter',
            color=preprocessing.LabelEncoder().fit_transform(target),
            colorscale='Portland',
            line=dict(color='rgb(255, 255, 255)'),
            opacity=0.9,
            size=2  # ごちゃごちゃしないように小さめに設定するのがオススメです．
        )
    )

    data = [trace1]
    layout = dict(height=700, width=600, title='coil-20 tsne exmaple')
    fig = dict(data=data, layout=layout)
    offline.plot(fig, filename='estimate_visualize/tsne_example', auto_open=True)


def plotly_t_sne(latent_vecs, target=None, data_type='test', dir_type=None):
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    fig = go.Figure(data=[
        go.Scatter(x=latent_vecs_reduced[:, 0], y=latent_vecs_reduced[:, 1], mode='markers+text',
                   marker=dict(colorscale='jet',

                               color=target),
                   text=target, )
    ])
    fig.write_html('result/tsne/t-SNE_{}_{}.html'.format(data_type, dir_type))
    # fig.show()


def t_sne(latent_vecs, target=None, data_type='test', dir_type=None):
    # latent_vecs = latent_vecs.to("cpu")
    # latent_vecs = latent_vecs.detach().numpy()
    start_time = time.time()
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    # latent_vecs_reduced = PCA(n_components=2).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                c=target, cmap='jet')
    plt.colorbar()
    plt.title('t-SNE {}_{}'.format(data_type, dir_type))
    plt.savefig('result/tsne/t-SNE_{}_{}.png'.format(data_type, dir_type))
    interval = time.time() - start_time
    print('t-SNE : {}s'.format(interval))
    plt.show()
    plt.close()


def show_umap(latent_vecs, target, data_type='test'):
    latent_vecs = latent_vecs.to("cpu")
    latent_vecs = latent_vecs.detach().numpy()
    start_time = time.time()
    embedding = umap.UMAP().fit_transform(latent_vecs)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='jet')
    plt.colorbar()
    plt.title('umap {}{}'.format(data_type, start_time))
    plt.savefig('result/umap/umap_{}{}.png'.format(data_type, start_time))
    interval = time.time() - start_time
    print('umap : {}s'.format(interval))
    plt.show()


if __name__ == '__main__':
    main()
