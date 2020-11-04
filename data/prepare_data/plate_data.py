import pandas as pd
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt
# pd.set_option('display.max_colwidth',1000)


class PlateData:
    def __init__(self, blank_node_csv):

        with open(blank_node_csv, encoding="shift-jis") as f:
            print(blank_node_csv)
            node_file_raw = pd.read_csv(f, encoding="shift-jis")
            node_file = node_file_raw[3:]
            node_file.columns = ['node_id', 'x', 'y', 'z']
        self.node_file = node_file
        self.conters_data = {}
        self.shell_origin = None

    def _read_conter_file(self, conter_csv_path):
        with open(conter_csv_path, encoding="shift-jis") as f:
            conter_raw = pd.read_csv(f, header=3, encoding="shift-jis")
            conter_file = conter_raw.drop("Unnamed: 5", axis=1)
            conter_file.columns = ['node_id', 'conter_value', 'x', 'y', 'z']
        return conter_file

    def set_dynain_data(self, dynain_data):
        # ['shell_id', 'normal_vector', 'x', 'y', 'z'] の形式 でshellの個数分の配列をshell_origin として保存
        # (x,y,z）はblankを採用
        shell_origin = []
        node_file_matrix = self.node_file.values
        node_dict = {node[0]: node for node in node_file_matrix}
        print(len(list(node_dict.keys())))
        for shell in dynain_data.shells:
            # ['shell_id', 'normal_vector', 'x', 'y', 'z'] の形式
            nodes = shell.nodes

            (x, y, z) = np.average([node_dict[str(node.node_id)].astype('float64')[1:4] for node in nodes], axis=0)

            shell_origin.append([shell.shell_id, shell.normal_vector, x, y, z])

        self.shell_origin = shell_origin

    def output(self, figsize=(16, 16)):
        # 形状の情報
        # conterの情報
        #         print(self.node_file)
        print(self.conters_data.keys())
        ret_array = self.get_normal_vector_fig(figsize)

        for key in self.conters_data.keys():
            conter_array = self.get_conter_fig(key)
            print(conter_array.shape)
            ret_array = np.concatenate([ret_array, conter_array], axis=2)

        print(ret_array.shape)
        return ret_array

    def output_labels(self):

        label_dict = {}
        print(self.conters_data.keys())
        for conter_name, conter in self.conters_data.items():
            label_dict[conter_name] = int(bool(self._contain_error(conter_name, conter)))

        print(label_dict)
        return label_dict

    @staticmethod
    def _contain_error(name, conter):
        if name == '2.板厚減少率CSV':
            error_num = PlateData.check_conter_value(conter, '>', 0.08)
            error_num += PlateData.check_conter_value(conter, '<', -0.1)
            print(error_num)

            return error_num

        # elif name == '3.シワコンターCSV':
        #     error_num = PlateData.check_conter_value(conter, '>', 0.05)
        #     error_num += PlateData.check_conter_value(conter, '<', -0.05)
        #     print(error_num)
            # return error_num

        else:
            return 0

    @staticmethod
    def check_conter_value(conter, operator, value):
        if operator == '>':
            sum_of_error = (conter['conter_value'] > value).sum()
        elif operator == '<':
            sum_of_error = (conter['conter_value'] < value).sum()
        return sum_of_error

    def fig2array(self, fig):

        buf = io.BytesIO()  # インメモリのバイナリストリームを作成
        # matplotlibから出力される画像のバイナリデータをメモリに格納する. todo set in config
        fig.savefig(buf, format="jpeg", dpi=16)
        buf.seek(0)  # ストリーム位置を先頭に戻る
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # メモリからバイナリデータを読み込み, numpy array 形式に変換
        buf.close()  # ストリームを閉じる(flushする)
        img = cv2.imdecode(img_arr, 1)  # 画像のバイナリデータを復元する
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.imread() はBGR形式で読み込むのでRGBにする.
        return img

    def get_normal_vector_fig(self, figsize=(16, 16)):
        shell_t = np.array(self.shell_origin).T
        value = shell_t[1] / 2 + 0.5
        x = shell_t[2]
        y = shell_t[3]

        fig = plt.figure(figsize=figsize, linewidth=0, )

        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        ax.scatter(x, y, c=value)
        ax.set_aspect('equal')
        ret_array = self.fig2array(fig)
        plt.close(fig)
        return ret_array

    def get_conter_fig(self, conter_name, figsize=(16, 16)):

        conter_data = self.get_plate_conter(conter_name)
        # todo conterの値を画像に反映する必要がある
        value = conter_data[1]
        x = conter_data[2]
        y = conter_data[3]

        fig = plt.figure(figsize=figsize)

        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        ax.scatter(x, y, c=value, vmin=np.min(value), vmax=np.max(value))
        ax.set_aspect('equal')
        ret_array = self.fig2array(fig)
        plt.close(fig)
        return ret_array

    def set_conter(self, conter_name, conter_csv_path):
        self.conters_data[conter_name] = self._read_conter_file(conter_csv_path)

    # pandas のraw dataのconterを返す
    def get_conter(self, conter_name):
        return self.conters_data[conter_name]

    def get_plate_conter(self, conter_name):
        # node_id, conter_data, x, y, z

        data = [self.node_file['node_id'].values,
                self.conters_data[conter_name]['conter_value'].values.astype('float64'),
                self.node_file['x'].values.astype('float64'),
                self.node_file['y'].values.astype('float64'),
                self.node_file['z'].values.astype('float64'),
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

    def plot_normal_vecotr(self, figsize=(16, 16), save_name=None):
        shell_t = np.array(self.shell_origin).T
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
    def __init__(self, node_id, x, y, z, translational_flag, rotate_flag):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.z = z
        self.translational_flag = translational_flag
        self.rotate_flag = rotate_flag
        self.belog_shells = []

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


class DynainData:
    def __init__(self, file_path):
        with open(file_path) as f:
            df = pd.read_csv(f)
        matrix = df.values
        asterisk_indexes = list(df[df['*KEYWORD'].str.startswith('*')].index)
        #         thickness_index = df[df['*KEYWORD']=='*ELEMENT_SHELL_THICKNESS'].index.values[0]
        #         initial_atress_index = df[df['*KEYWORD']=='*INITIAL_STRESS_SHELL'].index.values[0]

        node_matrix = matrix[asterisk_indexes[0] + 1:asterisk_indexes[1]]

        self.nodes = {int(node_data[0][0:8]): Node(*self.split_node_str(node_data[0])) for node_data in node_matrix}

        shell_data = matrix[asterisk_indexes[1] + 1:asterisk_indexes[2]]

        self.shells = [Shell(*self.split_shell_str_a(shell_data[i][0]), *self.split_shell_str_b(shell_data[i + 1][0]))
                       for i in range(0, len(shell_data), 2)]

        # todo それ以外のやつはまだ

    def split_node_str(self, node_str):
        return int(node_str[0:8]), float(node_str[8:24]), float(node_str[24:40]), float(node_str[40:56]), \
               node_str[63:64], node_str[-1:]

    def split_shell_str_a(self, shell_str):
        shell_nodes = [self.nodes[node_id] for node_id in
                       [int(shell_str[16:24]), int(shell_str[24:32]), int(shell_str[32:40]), int(shell_str[40:48])]]
        return int(shell_str[0:8]), int(shell_str[8:16]), shell_nodes

    def split_shell_str_b(self, node_str):
        return float(node_str[0:16]), float(node_str[16:32]), float(node_str[32:48]), float(node_str[48:64])
