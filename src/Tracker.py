import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output,
                                                          'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        """
        主要用于在批处理中进行一次相机参数的优化迭代。函数用于样本采样、渲染深度和颜色、计算损失，并执行反向传播以更新相机参数。
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W  # 默认20
        Hedge = self.ignore_edge_H  # # 默认20

        # 根据相机位姿和图像尺寸，生成一批射线的原点和方向，并获取对应的真实深度和彩色图像。
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H - Hedge, Wedge, W - Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        # 使用提供的相机参数和射线信息，对采样点进行渲染得到深度、不确定性和颜色
        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o, self.device, stage='color', gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()  # 使其不再与后续计算产生梯度关联
        if self.handle_dynamic:  # 是否动态处理深度图像的不确定性，默认 True
            tmp = torch.abs(batch_gt_depth - depth) / torch.sqrt(uncertainty + 1e-10)
            mask = (tmp < 10 * tmp.median()) & (batch_gt_depth > 0)  # 掩码用于在后续的优化中选择有效的深度值
        else:
            mask = batch_gt_depth > 0  # 直接使用深度图像作为掩码

        loss = (torch.abs(batch_gt_depth - depth) /
                torch.sqrt(uncertainty + 1e-10))[mask].sum()  # 损失是深度误差与不确定性的标准差之间的比率

        if self.use_color_in_tracking:  # 默认True
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss * color_loss  # 计算颜色误差

        loss.backward()  # 反向传播
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        print("Tracker.py_def run()")
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]  # 当前帧的索引
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            # 选择同步方法，默认strict
            if self.sync_method == 'strict':
                # 严格等待之前的映射完成
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx - 1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx - 1].to(device)
            elif self.sync_method == 'loose':
                # 达成while条件即可
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx - self.every_frame - self.every_frame // 2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # 无需等待，并行执行
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping()  # 根据同步方法更新参数

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ", idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w  # 使用真实相机位姿进行跟踪
                if not self.no_vis_on_first_frame:  # 如果不需要在第一帧进行可视化
                    # 调用self.visualizer.vis()函数进行目标跟踪的可视化
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)

            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                if self.const_speed_assumption and idx - 2 >= 0:  # 如果使用了恒定速度假设（默认True）并且当前帧的索引大于等于2
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w @ self.estimate_c2w_list[idx - 2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta @ pre_c2w  # 计算新的相机位姿
                else:
                    estimated_new_cam_c2w = pre_c2w  # 计算新的相机位姿

                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())  # 转换为相机张量
                if self.seperate_LR:  # 分离学习率，对相机张量的参数进行分离和优化，默认False
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr * 0.2}])
                else:  # 直接对整个相机张量进行优化
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device) - camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss = 10000000000.

                for cam_iter in range(self.num_cam_iters):  # 迭代优化相机位姿
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)  # 进行可视化

                    # 调用self.optimize_cam_in_batch()函数进行一次相机优化迭代，并得到当前优化的损失值loss
                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)

                    # 在第一次迭代时记录初始损失initial_loss
                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device) - camera_tensor).mean().item()  # 计算相机估计值与真实相机位姿之间的差
                    if self.verbose:
                        if cam_iter == self.num_cam_iters - 1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:  # 迭代loss
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()

                # 将相机位姿转换为齐次坐标形式
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)

            self.estimate_c2w_list[idx] = c2w.clone().cpu()  # 将当前帧的估计相机位姿保存
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()  # 将当前帧的真实相机位姿保存
            pre_c2w = c2w.clone()  # 将当前帧的估计相机位姿c2w保存到pre_c2w中，用于后续的下一帧跟踪时作为上一帧的相机位姿
            self.idx[0] = idx  # 更新目标跟踪算法的当前帧索引self.idx为当前帧的索引idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
