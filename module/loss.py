import math
import torch
import torch.nn as nn
from typing import List
# from utils import anchor_to_box_transform

class DetectorLoss(nn.Module):
    def __init__(self, anchor_sizes: List, device):
        super().__init__()
        self.device = device
        self.anchors = torch.tensor(anchor_sizes, device=self.device)

    def iou(self, box1, box2, eps=1e-7) -> torch.Tensor:
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box1 = box1.t()
        box2 = box2.t()

        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        return iou

    def s_iou(self, box1, box2, eps=1e-7) -> torch.Tensor:
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box1 = box1.t()
        box2 = box2.t()

        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
            1 - torch.exp(-1 * omiga_h), 4
        )
        iou = iou - 0.5 * (distance_cost + shape_cost)

        return iou

    def build_target(self, preds, targets):
        """
        生成gt_box、gt_cls、ps_index的过程中只用到了特征图的W, H
        """
        pred_obj, *_ = preds
        *_, H, W = pred_obj.shape

        # 每个网格的四个顶点为box中心点会归的基准点
        quadrant = torch.tensor([[0, 0], 
                                 [1, 0], 
                                 [0, 1], 
                                 [1, 1]], device=self.device)

        if targets.shape[0] == 0:
            gt_box, gt_cls, pos_index = None, None, None
            return gt_box, gt_cls, pos_index

        # 将坐标映射到特征图尺度上
        scale = torch.tensor([1, 1, W, H, W, H]).to(self.device)
        # scale[2:] = torch.tensor(preds.shape)[[3, 2, 3, 2]]
        # scale = [1, 1, W, H, W, H]
        gt: torch.Tensor = targets * scale # 把gt放大到特征图的尺寸
        # gt.shape = [N, 6]

        box_center: torch.Tensor = gt[:, 2:4]  # 切片找到中心点坐标
        box_center = box_center.repeat_interleave(4, dim=0)  # 复制四份
        box_center = box_center.long()  # 取整

        quadrant = quadrant.repeat(gt.size(0), 1)  # 把quadrant复制gt的数量
        box_center4: torch.Tensor = (
            box_center + quadrant
        )  # 将两个相加，变成四个相邻的像素点
        # breakpoint()
        box_center4 = box_center4.repeat_interleave(
            self.anchors.size(0), dim=0
        )  # 把这四个相邻的像素点复制9份
        anchors = self.anchors * torch.tensor([W, H]).to(self.anchors)  # 把anchor放大到特征图大小
        anchors = anchors.repeat(box_center.size(0), 1)  # 把anchor复制中心点的数量
        proposals = torch.cat(
            (box_center4, anchors), 1
        )  # 把中心点与框拼接在一起用来计算iou
        gt_repeat_4a = gt.repeat_interleave(4 * self.anchors.size(0), dim=0)  # 将gt复制为36行
        iou = self.iou(gt_repeat_4a[:, 2:], proposals)  # 计算iou
        iou = iou.reshape(-1, 4 * self.anchors.size(0))  # 每36个数据为一行
        iou_mask = iou > 0.7
        iou_nogt07 = torch.all(
            ~iou_mask, dim=1
        )  # 找是否整行都小于0.7，如果有，则这一行为true
        iou_nogt07_row_id = torch.argmax(
            iou[iou_nogt07], dim=1
        )  # 寻找整行都小于0.7时最大的那个
        # 选中整行都小于0.7中最大的那个为true
        iou_mask[iou_nogt07, iou_nogt07_row_id] = True
        iou_mask = iou_mask.reshape(-1)

        # TODO: 应该先用in_bound_mask滤一遍，再用iou_mask
        in_bound_mask = torch.logical_and(
            box_center4 < torch.tensor([W, H]).to(box_center4), 
            box_center4 >= torch.tensor([0, 0]).to(box_center4)
        ).all(dim=1)

        anchor_idx = torch.arange(
            start=0, end=self.anchors.size(0), step=1, dtype=torch.long, device=self.device
        ).repeat(gt.size(0) * 4)

        positive_anchor_idx = anchor_idx[iou_mask & in_bound_mask]
        positive_batch_idx = gt_repeat_4a[iou_mask & in_bound_mask, 0].long()
        positive_x_idx, positive_y_idx = box_center4[iou_mask & in_bound_mask].long().T

        """
        # 扩展维度复制数据，将一个网格变成其周围的4个网格
        gt = gt.repeat(4, 1, 1)
        # gt.shape = [4, N, 6]

        # 过滤越界坐标
        quadrant = quadrant[:, None, ...]
        quadrant = quadrant.repeat_interleave(gt.size(1), dim=1)
        # quadrant = quadrant.repeat(gt.size(1), 1, 1).permute(1, 0, 2)
        # quadrant.shape = [4, N, 2]
        # gt[..., 2:4]是box中心点的坐标，gij是取了中心点坐标的周围4个整数坐标
        # gij坐标的顺序是w, h
        gij = gt[..., 2:4].long() + quadrant

        # j = torch.where(gij < H, gij, 0).min(dim=-1)[0] > 0
        j = torch.logical_and(
                gij < torch.tensor([W, H]).to(gij),
                gij >= torch.tensor([0, 0]).to(gij)
            )
        # j.shape = [4, N, 2]
        j = torch.all(j, dim=-1)
        # j.shape = [4, N]

        # 前景的位置下标
        # 也就是说，要找到布尔数组索引j，使得j处的gij是在图片范围内的
        # gx为横坐标，gy为纵坐标，**都是一维矩阵**
        # 注意，此处的W，H是preds的WH，说明坐标就是特征图的下标
        gx, gy = gij[j].T
        batch_index = gt[..., 0].long()[j]
        # ps_index代表某张图的哪几个方格应该检测到物体
        pos_index.append((batch_index, gx, gy))
        """

        positive_index = (positive_batch_idx, positive_x_idx,
                          positive_y_idx, positive_anchor_idx)

        # 前景的box
        # 我发现用布尔矩阵切片，会导致布尔矩阵维度及之前的维度都会被压缩成一维
        gt_box = gt_repeat_4a[..., 2:][iou_mask & in_bound_mask]

        # 前景的类别
        gt_cls = gt_repeat_4a[..., 1].long()[iou_mask & in_bound_mask]

        return gt_box, gt_cls, positive_index

    def forward(self, preds, targets):
        # 初始化loss值
        ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
        cls_loss, iou_loss, obj_loss, tot_loss = ft([0]), ft([0]), ft([0]), ft([0])
        del ft

        # 定义obj和cls的损失函数
        # 确实是交叉熵，后面log了
        NLLloss_cls = nn.NLLLoss() 
        # smmoth L1相比于bce效果最好
        smoothL1_obj = nn.SmoothL1Loss(reduction='none')

        # 构建ground truth
        gt_box, gt_cls, positive_index = self.build_target(preds, targets)

        if gt_box is None:
            return iou_loss, obj_loss, cls_loss, tot_loss

        pred_obj, pred_delta_box, pred_cls = preds
        # breakpoint()
        *_, H, W = pred_obj.shape
        # to H,W,C
        pred_obj = pred_obj.permute(0, 2, 3, 1)
        # from B, A, 4, H, W to B, H, W, anchor_num, channel
        pred_delta_box = pred_delta_box.permute(0, 3, 4, 1, 2)
        pred_cls = pred_cls.permute(0, 3, 4, 1, 2)

        target_obj = torch.zeros_like(pred_obj)
        # factor应该是用来平衡正负样本的
        factor = torch.ones_like(pred_obj) * 0.75

        # 计算检测框回归loss
        p_batch_idx, p_x_idx, p_y_idx, p_anchor_idx = positive_index
        # breakpoint()
        pred_box = torch.zeros(
            pred_delta_box[p_batch_idx, p_y_idx, p_x_idx, p_anchor_idx].shape
        ).to(self.device)
        # breakpoint()
        pred_box[:, 0] = pred_delta_box[p_batch_idx, p_y_idx, 
                                        p_x_idx, p_anchor_idx][:, 0].tanh() + p_x_idx
        pred_box[:, 1] = pred_delta_box[p_batch_idx, p_y_idx, 
                                        p_x_idx, p_anchor_idx][:, 1].tanh() + p_y_idx
        pred_box[:, 2:4] = (
            torch.exp(pred_delta_box[p_batch_idx, p_y_idx, 
                                     p_x_idx, p_anchor_idx][:, 2:4]) 
            * self.anchors[p_anchor_idx] * torch.tensor([W, H], device=self.device)
        )

        # 计算检测框IOU loss
        iou = self.s_iou(pred_box, gt_box)
        """
        # Filter
        pos_iou_filter = iou > iou.mean()

        # breakpoint()
        # b, gx, gy变为正样本的数据
        # b, gy, gx = b[pos_iou_filter], gy[pos_iou_filter], gx[pos_iou_filter]
        p_batch_idx = p_batch_idx[pos_iou_filter]
        p_x_idx = p_x_idx[pos_iou_filter]
        p_y_idx = p_y_idx[pos_iou_filter]
        p_anchor_idx = p_anchor_idx[pos_iou_filter]

        # 计算iou loss
        iou = iou[pos_iou_filter]
        """
        iou_loss =  (1.0 - iou).mean() 

        # 计算目标类别分类分支loss
        pred_cls_log = torch.log(pred_cls[p_batch_idx, p_y_idx, p_x_idx, p_anchor_idx])
        """
        cls_loss = NLLloss_cls(pred_cls_log, gt_cls[pos_iou_filter])
        """
        cls_loss = NLLloss_cls(pred_cls_log, gt_cls)

        # iou aware
        target_obj[p_batch_idx, p_y_idx, p_x_idx, p_anchor_idx] = iou.float()
        # 统计每个图片正样本的数量
        # n[x]表示x在b中出现的次数
        n = torch.bincount(p_batch_idx)
        # 正样本越多，正样本对应的权重越小。
        # TODO: 直接换成focal loss会怎么样？
        factor[p_batch_idx, p_y_idx, p_x_idx, p_anchor_idx] = (1. / (n[p_batch_idx] / (H * W))) * 0.25

        # 计算前背景分类分支loss
        obj_loss = (smoothL1_obj(pred_obj, target_obj) * factor).mean()

        # 计算总loss
        tot_loss = (iou_loss) + (obj_loss * 4) + cls_loss * 2             

        return iou_loss, obj_loss, cls_loss, tot_loss
