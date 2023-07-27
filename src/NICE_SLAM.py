import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')


class NICE_SLAM():
    """
    NICE_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args
        self.nice = args.nice

        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        print("NICE-SLAM.py: 重置相机内参")
        self.update_cam()  # 重置相机内参

        # 获取网络
        print("NICE-SLAM.py: 获取网络")
        model = config.get_model(cfg,  nice=self.nice)
        self.shared_decoders = model

        self.scale = cfg['scale']

        print("NICE-SLAM.py: 初始化多层特征网格")
        self.load_bound(cfg)  # 加载场景边界参数（bound）并将其传递给不同的解码器（decoders）和当前对象（self）
        if self.nice:
            self.load_pretrain(cfg)  # 加载预训练的ConvOnet模型权重参数到解码器（decoders）中
            self.grid_init(cfg)  # 初始化 多层特征网格 hierarchical feature grids 并存储到 self.shared_c 中
        else:
            self.shared_c = {}

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        print("NICE-SLAM.py: 载入数据集")
        self.frame_reader = get_dataset(cfg, args, self.scale)  # 载入数据集
        self.n_img = len(self.frame_reader)  # 计算数据集中图片的数量
        print("NICE-SLAM.py: {}".format(self.n_img))

        print("NICE-SLAM.py: 初始化列表")
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))  # 创建了一个 self.n_img * 4 * 4 的 tensor： estimate_c2w_list
        self.estimate_c2w_list.share_memory_()  # 将c2w的估计值列表 estimate_c2w_list 设置为共享内存

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))  # 创建了一个 self.n_img * 4 * 4 的 tensor： gt_c2w_list
        self.gt_c2w_list.share_memory_()  # 将c2w的gt值列表 estimate_c2w_list 设置为共享内存

        self.idx = torch.zeros((1)).int()  # 创建了一个单个零值的 tensor： idx索引
        self.idx.share_memory_()  # 设置为共享内存

        self.mapping_first_frame = torch.zeros((1)).int()  # mapping_first_frame 表示 Mapper 中最新帧的索引
        self.mapping_first_frame.share_memory_()

        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()  # mapping_idx 用于表示 Mapper 中当前帧的索引
        self.mapping_idx.share_memory_()

        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping； mapping 计数器
        self.mapping_cnt.share_memory_()

        for key, val in self.shared_c.items():  # 该遍历主要用来将特征网格的值 val 转移到 device 上
            val = val.to(self.cfg['mapping']['device'])
            val.share_memory_()
            self.shared_c[key] = val

        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])  # 将共享解码器 shared_decoders 转移到 device 上
        self.shared_decoders.share_memory()

        self.renderer = Renderer(cfg, args, self)  # 初始化渲染器
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)  # 初始化日志器

        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)  # 初始化 mapper
        if self.coarse:
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True) # 初始化 coarse mapper
        self.tracker = Tracker(cfg, args, self)  # 初始化 tracker
        print("NICE-SLAM.py: 打印配置信息")
        self.print_output_desc()  # 打印配置信息

    def print_output_desc(self):  # 打印配置信息
        print("---打印配置信息---")
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")
        print("---打印结束---")

    def update_cam(self):
        """
        根据预处理配置更新相机内参
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:  # 配置中存在crop_size
            print("crop_size: True 需要对图像进行裁切")
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            print("crop_edge > 0  需要对图像边缘进行裁切")
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.
        加载场景边界参数（bound）并将其传递给不同的解码器（decoders）和当前对象（self）

        Args:
            cfg (dict): parsed config dict.  接收一个配置字典
        """
        # scale the bound if there is a global scaling factor
        # 将原始的场景边界参数cfg['mapping']['bound']乘以缩放因子self.scale(默认是1)后转为tensor
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound'])*self.scale)
        bound_divisible = cfg['grid_len']['bound_divisible']  # bound_divisible 默认为 0.32
        # enlarge the bound a bit to allow it divisible by bound_divisible
        # 使其能够被bound_divisible整除
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisible).int()+1)*bound_divisible+self.bound[:, 0]
        if self.nice:  # 将缩放后的边界传递给shared_decoders的不同网格
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound*self.coarse_bound_enlarge

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.
        加载预训练的ConvOnet模型权重参数到解码器（decoders）中

        Args:
            cfg (dict): parsed config dict
        """

        if self.coarse:  # 加载粗糙网格的权重并存入运行设备
            ckpt = torch.load(cfg['pretrained_decoders']['coarse'],
                              map_location=cfg['mapping']['device'])
            coarse_dict = {}  # 用于存储粗糙解码器的权重参数
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            # 将存储在coarse_dict中的粗糙解码器的权重参数加载到
            # self.shared_decoders.coarse_decoder中，确保该解码器具有预训练的权重
            self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict)

        # 加载中间级网格的权重并存入运行设备
        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}  # 用于存储中间级解码器的权重参数
        fine_dict = {}  # 用于存储景精细级解码器的权重参数
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val

        #  与粗糙级类似
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict)
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict)

    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids.
        初始化 多层特征网格（hierarchical feature grids）

        Args:
            cfg (dict): parsed config dict.
        """

        #  从配置文件加载特征网格的网格尺寸
        if self.coarse:
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        c = {}  # 空字典 c 用于存储不同层次特征网格的值
        c_dim = cfg['model']['c_dim']  # c_dim = 32
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        # If you have questions regarding the swap of axis 0 and 2,
        # please refer to https://github.com/cvg/nice-slam/issues/24

        if self.coarse:
            coarse_key = 'grid_coarse'
            coarse_val_shape = list(
                map(int, (xyz_len*self.coarse_bound_enlarge/coarse_grid_len).tolist()))
            coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]
            self.coarse_val_shape = coarse_val_shape
            val_shape = [1, c_dim, *coarse_val_shape]
            coarse_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
            c[coarse_key] = coarse_val

        middle_key = 'grid_middle'  # 用于存储中间级网格的键
        middle_val_shape = list(map(int, (xyz_len/middle_grid_len).tolist()))  # 计算中间网格的尺寸 middle_val_shape
        middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]  # 交换 X 和 Z 轴
        self.middle_val_shape = middle_val_shape  # 保存网格的尺寸信息
        val_shape = [1, c_dim, *middle_val_shape]  # val_shape 包含了中间级网格的形状信息
        # 创建一个具有相应形状的tensor middle_val，该张量初始化为均值为0，标准差为0.01的正态分布随机数
        middle_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[middle_key] = middle_val  # 将初始化的中间级网格 middle_val 存入 c 中

        fine_key = 'grid_fine'
        fine_val_shape = list(map(int, (xyz_len/fine_grid_len).tolist()))
        fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        self.fine_val_shape = fine_val_shape
        val_shape = [1, c_dim, *fine_val_shape]
        fine_val = torch.zeros(val_shape).normal_(mean=0, std=0.0001)  # 精细级特征网格的标准差更小
        c[fine_key] = fine_val

        color_key = 'grid_color'
        color_val_shape = list(map(int, (xyz_len/color_grid_len).tolist()))
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)  # 颜色特征网格的标准差
        c[color_key] = color_val

        self.shared_c = c

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while (1):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def coarse_mapping(self, rank):
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(3):
            if rank == 0:
                print("tracking")
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                print("mapping")
                p = mp.Process(target=self.mapping, args=(rank, ))
            elif rank == 2:
                if self.coarse:
                    print("coarse_mapping")
                    p = mp.Process(target=self.coarse_mapping, args=(rank, ))
                else:
                    continue
            p.start()
            processes.append(p)

        for p in processes:
            p.join()  # 并行执行三个函数


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
