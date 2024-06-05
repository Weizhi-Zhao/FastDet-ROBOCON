import yaml
import torch
import torchvision
from typing import Literal

# 解析yaml配置文件
class LoadYaml:
    def __init__(self, path):
        with open(path, encoding='utf8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.val_txt = data["DATASET"]["VAL"]
        self.train_txt = data["DATASET"]["TRAIN"]
        self.names = data["DATASET"]["NAMES"]
        self.anchor_sizes = data["DATASET"]["ANCHOR_SIZES"]

        self.learn_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.milestones = data["TRAIN"]["MILESTIONES"]
        self.end_epoch = data["TRAIN"]["END_EPOCH"]
        
        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]
        
        self.category_num = data["MODEL"]["NC"]
        
        print("Load yaml sucess...")

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 后处理(归一化后的坐标)
def handle_preds(preds, anchors, device, conf_thresh=0.25, nms_thresh=0.45):
    # anchors = torch.tensor(anchors, device=device)
    total_bboxes, output_bboxes  = [], []
    # 将特征图转换为检测框的坐标
    pred_obj, pred_reg, pred_cls = preds
    N, A, C, H, W = pred_reg.shape
    bboxes = torch.zeros((N, H, W, A, 6))
    # to (N, H, W, C)
    pred_obj = pred_obj.permute(0, 2, 3, 1)
    # to (N, H, W, A, C)
    pred_reg = pred_reg.permute(0, 3, 4, 1, 2)
    pred_cls = pred_cls.permute(0, 3, 4, 1, 2)

    # 检测框置信度
    # ... represents all other dimensions
    # confidence score
    bboxes[..., 4] = pred_obj.squeeze(-1) * 0.4 + pred_cls.max(dim=-1)[0] * 0.6
    # classification
    bboxes[..., 5] = pred_cls.argmax(dim=-1)

    # 检测框的坐标
    gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)])
    gy = gy.unsqueeze(2).repeat(1, 1, anchors.size(0))
    gx = gx.unsqueeze(2).repeat(1, 1, anchors.size(0))
    bw = torch.exp(pred_reg[..., 2]) * anchors[:, 0]
    bh = torch.exp(pred_reg[..., 3]) * anchors[:, 1]
    bw = bw.clamp(min=0.0, max=1)
    bh = bh.clamp(min=0.0, max=1)
    # breakpoint()
    bcx = (pred_reg[..., 0].tanh() + gx.to(device)) / W
    bcy = (pred_reg[..., 1].tanh() + gy.to(device)) / H

    # cx,cy,w,h = > x1,y1,x2,y1
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H*W*A, 6)
    total_bboxes.append(bboxes)
    # breakpoint()
    batch_bboxes = torch.cat(total_bboxes, 1)
    # breakpoint()
    # 对检测框进行NMS处理
    for p in batch_bboxes:
        temp = []
        output = []
        b, s, c = [], [], []
        # 阈值筛选
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # Torchvision NMS
        if len(b) > 0:
            b = torch.Tensor(b).to(device)
            c = torch.Tensor(c).squeeze(1).to(device)
            s = torch.Tensor(s).squeeze(1).to(device)
            # keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            # for i in keep:
            #     output.append(temp[i])
            keep = soft_nms(b, s, c)
            output = [temp[i] for i in keep]
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)

    max_top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    min_bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter_wh = (min_bottom_right - max_top_left).clamp(min=0)  # [N,M,2]  # 删除面积小于0 不相交的  clamp 钳；夹钳；
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N,M]  # 切片的用法 相乘维度减1
    iou = inter / (area1[:, None] + area2 - inter)
    '''
    >>> b = torch.tensor([2, 2, 3])
    >>> a = torch.tensor([[1], [10]])
    >>> a + b
    tensor([[ 3,  3,  4],
            [12, 12, 13]])
    '''
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    catergory: torch.Tensor,
    soft_threshold=0.8,
    iou_threshold=0.6,
    weight_method: Literal["gaussian", "linear"] = "gaussian",
    sigma=0.1,
):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（选取的得分TopK）之后的
    :param scores: [N]
    :param iou_threshold: 0.7
    :param soft_threshold soft nms 过滤掉得分太低的框 （手动设置）
    :param weight_method 权重方法 1. 线性 2. 高斯
    :return:
    """
    keep = []
    idxs = scores.argsort(descending=True)
    # while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
    for i in range(idxs.size(0)):
        # 由于scores得分会改变，所以每次都要重新排序，获取得分最大值
        idxs = scores.argsort(descending=True)  # 评分排序
        if i == idxs.size(0) - 1:  # 就剩余一个框了；
            keep.append(idxs[i])
            break
        keep_len = len(keep)
        max_score_index = idxs[i]
        keep.append(max_score_index)

        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        idxs = idxs[:-(keep_len + 1)]

        other_boxes_idx = torch.zeros(scores.size(0), dtype=torch.bool).to(scores.device)
        other_boxes_idx[idxs[i+1:]] = True
        other_boxes_idx = other_boxes_idx & (catergory == catergory[max_score_index])
        other_boxes = boxes[other_boxes_idx]  # [?, 4]

        
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        # Soft NMS 处理， 和 得分最大框 IOU大于阈值的框， 进行得分抑制
        if weight_method == "linear":   # 线性抑制  # 整个过程 只修改分数
            ge_threshod_bool = ious[0] >= iou_threshold
            ge_threshod_idxs = idxs[ge_threshod_bool]
            scores[ge_threshod_idxs] *= (1. - ious[0][ge_threshod_bool])  # 小于IoU阈值的不变
        elif weight_method == "gaussian":  # 高斯抑制， 不管大不大于阈值，都计算权重
            scores[other_boxes_idx] *= torch.exp(-(ious[0] * ious[0]) / sigma) # 权重(0, 1]

    keep = idxs.new(keep)  # Tensor
    keep = keep[scores[keep] > soft_threshold]  # 最后处理阈值
    # boxes = boxes[keep]  # 保留下来的框
    # scores = scores[keep]  # soft nms抑制后得分
    # return boxes, scores
    return keep
