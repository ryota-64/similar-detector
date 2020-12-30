import io
import itertools
import multiprocessing as multi
from multiprocessing import Pool

import pymesh
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from visualize.plot_data import ConterName


class MaxConter:
    SHAPE = 0.5
    THICKNESS_REDUCTION_RATE = 10
    WRINKLES = 0.13
    VON_MISES = 2000
    STRAIN = 0.3
    BASE_SHAPE = 0.5


class MinConter:
    SHAPE = - 0.5
    THICKNESS_REDUCTION_RATE = -10
    WRINKLES = - 0.13
    VON_MISES = 0
    STRAIN = 0
    BASE_SHAPE = - 0.5


class PlateData:
    def __init__(self, blank_node_csv):
        # encording = 'shift-jis'
        encording = 'ISO-8859-1'
        with open(blank_node_csv, encoding=encording) as f:
            print(blank_node_csv)
            node_file_raw = pd.read_csv(f, encoding=encording, header=3)
            node_file = node_file_raw
            node_file.columns = ['node_id', 'coord', 'x', 'y', 'z', 'unnamed']
            node_file = node_file.astype({'x': float, 'y': float, 'z': float})

        print('blank {} nodes'.format(node_file.values.shape))
        self.blank_node_file = node_file
        self.conters_data = {}
        self.shell_origin = None
        self.shell_origin_normal = None
        self.nodes = None  # set by dynain

        # 参照渡しになっているので、以下の２つのvalueは同じものが入るようになっている
        self.shells = []
        self.shells_dict = None

    def _read_conter_file(self, conter_csv_path):
        with open(conter_csv_path, encoding="shift-jis") as f:
            conter_raw = pd.read_csv(f, header=3, encoding="shift-jis")
            conter_file = conter_raw.drop("Unnamed: 5", axis=1)
            conter_file.columns = ['node_id', 'conter_value', 'x', 'y', 'z']
            # print('conter {} nodes '.format(conter_file.values.shape))
        return conter_file

    def __sub__(self, other):

        # todo blank_node_file が同一かのチェック

        # todo shell をそれぞれ差分とる

        pass

    # def set_dynain_data_old(self, dynain_data):
    #     # ['shell_id', 'normal_vector', 'x', 'y', 'z'] の形式 でshellの個数分の配列をshell_origin として保存
    #     # (x,y,z）はblankを採用
    #     shell_origin_normal = []
    #     node_file_matrix = self.blank_node_file.values
    #     node_dict = {node[0]: node for node in node_file_matrix}
    #     for shell in dynain_data.shells:
    #         # ['shell_id', 'normal_vector', 'x', 'y', 'z'] の形式
    #         nodes = shell.nodes
    #
    #         (x, y, z) = np.average([node_dict[str(node.node_id)].astype('float64')[1:4] for node in nodes], axis=0)
    #
    #         shell_origin_normal.append([shell.shell_id, shell.normal_vector, x, y, z])
    #
    #     self.shell_origin_normal = shell_origin_normal

    def set_dynain_data(self, dynain_data):
        # [shell_object, (x_min, x_max, y_min, y_max)]
        # の形式 でshellの個数分の配列をshell_origin として保存 (2つ目はblank_area)
        shell_origin = []
        self.blank_node_file = self.blank_node_file[:len(dynain_data.nodes)]
        # todo? input の形によらない( 間に座標に和があっても大丈夫なように）
        node_file_matrix = self.blank_node_file.values
        blank_node_dict = {str(int(node[0])): node for node in node_file_matrix}
        self.nodes = dynain_data.nodes

        for shell in dynain_data.shells:
            contain_nodes = [blank_node_dict[str(node.node_id)] for node in shell.nodes]
            shell.set_blank_nodes(contain_nodes)

            self.shells.append(shell)
            shell_origin.append([shell, shell.blank_area])
        self.shell_origin = shell_origin
        self.shells_dict = {str(int(shell.shell_id)): shell for shell in dynain_data.shells}
        # print(len(self.shell_origin))

    # 画像を出力する
    def output(self, output_size=(256, 256)):

        # blankのx, y の　max, min をだす
        x_max = float(max(self.blank_node_file['x']))
        x_min = float(min(self.blank_node_file['x']))
        y_max = float(max(self.blank_node_file['y']))
        y_min = float(min(self.blank_node_file['y']))
        print(x_max, x_min, y_max, y_min)
        x_range = np.linspace(min(x_min, y_min), max(x_max, y_max), output_size[0])
        y_range = np.linspace(min(x_min, y_min), max(x_max, y_max), output_size[1])
        xx, yy = np.meshgrid(x_range, y_range)
        worker_num = multi.cpu_count() - 2
        print('worker num: ', worker_num)
        p = Pool(worker_num)

        ret_array = p.starmap(self.pickup_shell_and_output, itertools.product(x_range, y_range))
        # ret_array = p.starmap(self.wrap, itertools.product(x_range, y_range))
        p.close()
        ret_array = np.array([np.array(line) for line in ret_array]).reshape((output_size[0], output_size[1], -1))
        print(ret_array.shape)
        print('a')

        return np.array(ret_array)

    def output_fig(self, figsize=(16, 16)):

        output_arrays = self.get_curvature_fig(figsize)[:, :, np.newaxis]
        for conter_key in self.conters_data.keys():
            print(output_arrays.shape)
            output_arrays = np.concatenate(
                [output_arrays, self.get_conter_fig(conter_key, figsize=figsize)[:, :, np.newaxis]], axis=2)

        output_arrays = np.concatenate([output_arrays, self.get_blank_fig(figsize=figsize)[:, :, np.newaxis]], axis=2)
        return np.array(output_arrays)

    # 出力したい画像のピクセルの位置のshellを取ってきて、各値を出力する（x,yを指定して取ってくる)
    def pickup_shell_and_output(self, x, y):

        candidate_shells = [shell[0] for shell in self.shell_origin
                            if shell[1][0] <= x <= shell[1][1]
                            and shell[1][2] <= y <= shell[1][3]]
        pickup_shell = [shell for shell in candidate_shells if shell.is_contain_point(x, y)]
        if pickup_shell:
            # 複数あった場合は一個目のものを採用
            return pickup_shell[0].output()
        else:
            return np.zeros_like(self.shells[0].output())

    def fig2array(self, fig, color=False):

        buf = io.BytesIO()  # インメモリのバイナリストリームを作成
        # matplotlibから出力される画像のバイナリデータをメモリに格納する. todo set in config
        fig.savefig(buf, format="jpeg", dpi=16)
        buf.seek(0)  # ストリーム位置を先頭に戻る
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # メモリからバイナリデータを読み込み, numpy array 形式に変換
        buf.close()  # ストリームを閉じる(flushする)
        img = cv2.imdecode(img_arr, 1)  # 画像のバイナリデータを復元する
        if color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.imread() はBGR形式で読み込むのでRGBにする.
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def get_normal_vector_fig(self, figsize=(16, 16)):
        shell_t = np.array(self.shell_origin).T
        value = shell_t[1] / 2 + 0.5
        x = shell_t[2]
        y = shell_t[3]

        fig = plt.figure(figsize=figsize, linewidth=0, )

        ax = fig.add_axes((0, 0, 1, 1))
        # ax.axis("off")
        ax.scatter(x, y, c=value)
        ax.set_aspect('equal')
        ret_array = self.fig2array(fig)
        plt.close(fig)
        return ret_array

    def get_blank_fig(self, figsize=(16, 16)):
        x = []
        y = []
        for shell in self.shells:
            x.append((shell.blank_area[0] + shell.blank_area[1]) / 2)
            y.append((shell.blank_area[2] + shell.blank_area[3]) / 2)
        plt.style.use('dark_background')

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes((0, 0, 1, 1))

        ax.axis("off")
        # ax.scatter(x, y, c=curvature_data, vmin=np.min(curvature_data), vmax=np.max(curvature_data), cmap='binary')
        ax.scatter(x, y)
        ax.set_aspect('equal')
        ret_array = self.fig2array(fig)
        plt.close(fig)
        return ret_array

    def get_curvature_fig(self, figsize=(16, 16)):

        curvature_data = []
        x = []
        y = []
        for shell in self.shells:
            curvature_data.append(shell.gaussian_curvature)
            x.append((shell.blank_area[0] + shell.blank_area[1]) / 2)
            y.append((shell.blank_area[2] + shell.blank_area[3]) / 2)
        plt.style.use('dark_background')
        fig = plt.figure(figsize=figsize)

        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        # ax.scatter(x, y, c=curvature_data, vmin=np.min(curvature_data), vmax=np.max(curvature_data), cmap='binary')
        ax.scatter(x, y, c=curvature_data, vmin=MinConter.SHAPE, vmax=MaxConter.SHAPE, cmap='gray')
        ax.set_aspect('equal')
        ret_array = self.fig2array(fig)
        plt.close(fig)
        return ret_array

    def get_conter_fig(self, conter_name, figsize=(16, 16)):
        print(conter_name)
        print(conter_name.split('.')[0].split('_')[-1])
        conter_enum_name = str(ConterName(int(conter_name.split('.')[0].split('_')[-1]))).split('.')[1]
        print(conter_enum_name)
        conter_data = self.get_plate_conter(conter_name)
        # todo conterの値を画像に反映する必要がある
        value = conter_data[1]
        x = conter_data[2]
        y = conter_data[3]

        plt.style.use('dark_background')
        fig = plt.figure(figsize=figsize)

        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        # ax.scatter(x, y, c=value, vmin=np.min(value), vmax=np.max(value), cmap='binary')
        ax.scatter(x, y, c=value, vmin=getattr(MinConter, conter_enum_name), vmax=getattr(MaxConter, conter_enum_name), cmap='gray')
        ax.set_aspect('equal')
        ret_array = self.fig2array(fig)
        plt.close(fig)
        return ret_array

    def set_conter(self, conter_name, conter_csv_path):
        self.conters_data[conter_name] = self._read_conter_file(conter_csv_path)
        # print(self.conters_data[conter_name].values.shape)
        for row in self.conters_data[conter_name].values:
            try:
                self.shells_dict[str(int(row[0]))].conter_values[conter_name] = row[1]
            except:
                continue

        # todo shellにconter値を入れる

    # pandas のraw dataのconterを返す
    def get_conter(self, conter_name):
        return self.conters_data[conter_name]

    def get_plate_conter(self, conter_name):
        # node_id, conter_data, x, y, z
        #
        # data = [self.blank_node_file['node_id'].values,
        #         self.conters_data[conter_name]['conter_value'].values.astype('float64'),
        #         self.blank_node_file['x'].values.astype('float64'),
        #         self.blank_node_file['y'].values.astype('float64'),
        #         self.blank_node_file['z'].values.astype('float64'),
        #         ]
        #
        # todo conterの value部分が要素に変わっているので注意 画像の出力時。多分下のやつでいけてるはず
        data = [self.conters_data[conter_name]['node_id'],
                self.conters_data[conter_name]['conter_value'].values.astype('float64'),
                [np.average(
                    [float(node[2]) for node in
                     self.shells_dict[str(int(self.conters_data[conter_name]['node_id'][i]))].blank_nodes]) for i in
                    range(len(self.conters_data[conter_name]['node_id']))],

                [np.average(
                    [float(node[3]) for node in
                     self.shells_dict[str(int(self.conters_data[conter_name]['node_id'][i]))].blank_nodes]) for i in
                    range(len(self.conters_data[conter_name]['node_id']))],
                [np.average(
                    [float(node[4]) for node in
                     self.shells_dict[str(int(self.conters_data[conter_name]['node_id'][i]))].blank_nodes]) for i in
                    range(len(self.conters_data[conter_name]['node_id']))],
                ]
        return data

    # PlateDataのインスタンス同士でのnormal_vectorの引き算
    @staticmethod
    def calc_normal_vector_diff(p1, p2, figsize=(16, 16), save_name=None):
        shell_t = np.array(p1.shell_origin).T
        p1_value = shell_t[1] / 2 + 0.5
        x = shell_t[2]
        y = shell_t[3]

        p2_shell_t = np.array(p2.shell_origin).T
        p2_value = p2_shell_t[1] / 2 + 0.5

        diff_value = (p1_value - p2_value) * 20 / 2 + 0.5

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, c=diff_value)
        ax.set_title('normal_vector_diff')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        print(diff_value)
        #         plt.colorbar(sc)
        if save_name:

            fig.savefig(save_name)
        else:
            plt.show()
        plt.close(fig)

        # PlateDataのインスタンス同士でのnormal_vectorの引き算

    @staticmethod
    def calc_conter_diff(p1, p2, conter_name, figsize=(16, 16), save_name=None):
        shell_t = np.array(p1.get_plate_conter(conter_name))
        p1_value = shell_t[1]
        x = shell_t[2]
        y = shell_t[3]

        p2_shell_t = np.array(p2.get_plate_conter(conter_name))
        p2_value = p2_shell_t[1]

        diff_value = (p1_value - p2_value)

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 1, 1)
        sc = ax.scatter(x, y, c=diff_value, vmin=np.min(diff_value), vmax=np.max(diff_value))
        ax.set_title('{}_diff'.format(conter_name))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        print(diff_value)
        plt.colorbar(sc)
        if save_name:
            fig.savefig(save_name)
        else:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_plate_conter(conter_data, figsize=(16, 16), save_name=None):

        value = conter_data[1]
        x = conter_data[2]
        y = conter_data[3]

        #         fig = plt.figure(figsize=(6,6))
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 1, 1)
        sc = ax.scatter(x, y, c=value, vmin=np.min(value), vmax=np.max(value))
        ax.set_title('conter plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(sc)

        if save_name:
            #             ax.set_rasterized(True)
            #             fig.savefig('test.pdf', dpi=300)
            fig.savefig(save_name)
        else:
            #             ax.set_rasterized(True)
            #             fig.savefig('test.pdf', dpi=300)
            plt.show()
        plt.close(fig)

    #         fig.savefig('big.png',format='png')

    def plot_normal_vector(self, figsize=(16, 16), save_name=None):
        shell_t = np.array(self.shell_origin_normal).T
        value = shell_t[1] / 2 + 0.5
        x = shell_t[2]
        y = shell_t[3]

        #         fig = plt.figure(figsize=(6,6))
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, c=value)
        ax.set_title('normal_vector')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        #         plt.colorbar(sc)
        if save_name:
            #             ax.set_rasterized(True)
            #             fig.savefig('test.pdf', dpi=400000)
            fig.savefig(save_name)
        else:
            ax.set_rasterized(True)
            plt.show()
        plt.close(fig)


class Node(object):
    def __init__(self, node_id, x, y, z, translational_flag, rotate_flag, gaussian_curvature):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.z = z
        self.translational_flag = translational_flag
        self.rotate_flag = rotate_flag
        self.gaussian_curvature = gaussian_curvature
        self.conter_values = {}
        self.belong_shells = []

    def get_points(self):
        return [self.x, self.y, self.z]

    def __repr__(self):
        return '<Node object node_id: {}>'.format(self.node_id)


class Shell:
    def __init__(self, shell_id, part, nodes, thick1, thick2, thick3, thick4):
        self.shell_id = shell_id
        self.part = part
        self.nodes = nodes
        self.thicknesses = [thick1, thick2, thick3, thick4]
        self.blank_nodes = []
        self.blank_area = None
        self.conter_values = {}

        # 代入されたnodesを使って、曲率を入れる
        self.gaussian_curvature = self.calc_shell_gaussian_curvature(nodes)
        self.normal_vector = self.create_normal_vector([node.get_points() for node in self.nodes][0:3])
        self.average_thickness = np.average([thick1, thick2, thick3, thick4])

    def create_normal_vector(self, points):
        p0, p1, p2 = points
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = [x1 - x0, y1 - y0, z1 - z0]
        vx, vy, vz = [x2 - x0, y2 - y0, z2 - z0]
        u_cross_v = np.array([uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]) / np.power(
            ((uy * vz - uz * vy) ** 2 + (uz * vx - ux * vz) ** 2 + (ux * vy - uy * vx) ** 2), 1 / 2)

        normal = np.array(u_cross_v)
        return normal

    def output(self):
        # x = np.average([float(node[1]) for node in self.blank_nodes])
        # y = np.average([float(node[2]) for node in self.blank_nodes])
        # z = np.average([float(node[3]) for node in self.blank_nodes])

        gaussian_curvature = self.gaussian_curvature
        conter_value_list = []
        for conter_name, value in self.conter_values.items():
            conter_value_list.append(value)
        # return np.array([x, y, z, gaussian_curvature, self.shell_id], dtype=np.float32)
        return np.array([gaussian_curvature, *conter_value_list], dtype=np.float32)

    def calc_shell_gaussian_curvature(self, nodes):
        gaussian_curvature = np.average([node.gaussian_curvature for node in nodes])
        if np.isnan(gaussian_curvature):
            gaussian_curvature = 0
        return gaussian_curvature

    def set_blank_nodes(self, blank_nodes):
        self.blank_nodes = blank_nodes
        x_max = float(max([node[2] for node in blank_nodes]))
        y_max = float(max([node[3] for node in blank_nodes]))
        x_min = float(min([node[2] for node in blank_nodes]))
        y_min = float(min([node[3] for node in blank_nodes]))
        self.blank_area = (x_min, x_max, y_min, y_max)

    def is_contain_point(self, x, y):
        if not self.blank_nodes:
            raise ValueError('Have not set origin nodes data')
        else:
            node_points = [node.astype('float64')[2:5] for node in self.blank_nodes]
            poly_points = [(point[0], point[1]) for point in node_points]
            polygon = Polygon(poly_points)
            point = Point(x, y)
            return polygon.contains(point)


class DynainData:
    def __init__(self, file_path):
        with open(file_path) as f:
            df = pd.read_csv(f)
        matrix = df.values
        asterisk_indexes = list(df[df['*KEYWORD'].str.startswith('*')].index)
        #         thickness_index = df[df['*KEYWORD']=='*ELEMENT_SHELL_THICKNESS'].index.values[0]
        #         initial_atress_index = df[df['*KEYWORD']=='*INITIAL_STRESS_SHELL'].index.values[0]

        node_matrix = matrix[asterisk_indexes[0] + 1:asterisk_indexes[1]]

        self.raw_nodes = [self.split_node_str(node_data[0])[1:4] for node_data in node_matrix]
        shell_data = matrix[asterisk_indexes[1] + 1:asterisk_indexes[2]]
        self.raw_shells = [self.split_shell_str_c(shell_data[i][0]) for i in range(0, len(shell_data), 2)]
        # pymeshを使って、曲率をだす
        vertices, faces = self._data_for_pymesh()
        mesh = pymesh.form_mesh(vertices, faces)
        triangle_mesh = pymesh.quad_to_tri(mesh)

        # triangle_mesh.add_attribute("vertex_gaussian_curvature")
        # nodes_gaussian_curvature = triangle_mesh.get_attribute("vertex_gaussian_curvature")
        triangle_mesh.add_attribute("vertex_mean_curvature")
        nodes_gaussian_curvature = triangle_mesh.get_attribute("vertex_mean_curvature")

        self.nodes = {str(int(node_data[0][0:8])): Node(*self.split_node_str(node_data[0]), gaussian_curvature)
                      for node_data, gaussian_curvature in zip(node_matrix, nodes_gaussian_curvature)}
        self.shells = [Shell(*self.split_shell_str_a(shell_data[i][0]), *self.split_shell_str_b(shell_data[i + 1][0]))
                       for i in range(0, len(shell_data), 2)]

        # print('dynain {} nodes'.format(len(node_matrix)))

    def _data_for_pymesh(self):
        return np.array(self.raw_nodes), np.array(self.raw_shells)

    # todo *KEYWORDの２つ目以降のやつはまだ
    def split_node_str(self, node_str):
        return int(node_str[0:8]), float(node_str[8:24]), float(node_str[24:40]), float(node_str[40:56]), \
               node_str[63:64], node_str[-1:]

    def split_shell_str_a(self, shell_str):
        shell_nodes = [self.nodes[str(node_id)] for node_id in
                       [int(shell_str[16:24]), int(shell_str[24:32]), int(shell_str[32:40]), int(shell_str[40:48])]]
        return int(shell_str[0:8]), int(shell_str[8:16]), shell_nodes

    def split_shell_str_b(self, node_str):
        return float(node_str[0:16]), float(node_str[16:32]), float(node_str[32:48]), float(node_str[48:64])

    # for pymesh nodes ids
    # dynain data's node id started from 1, but pymesh only use list. so index -1
    def split_shell_str_c(self, shell_str):
        return [int(shell_str[16:24]) - 1, int(shell_str[24:32]) - 1,
                int(shell_str[32:40]) - 1, int(shell_str[40:48]) - 1]
