import os
import cv2
import onnx
import time
import argparse
from onnxsim import simplify
from utils.datasets import *
import torch
from utils.tool import *
from module.detector import Detector
from module.loss import DetectorLoss
from tqdm import tqdm

if __name__ == "__main__":
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="", help=".yaml config")
    parser.add_argument("--weight", type=str, default=None, help=".weight config")
    parser.add_argument(
        "--img_folder", type=str, default="", help="The path of test image folder"
    )
    parser.add_argument(
        "--thresh", type=float, default=0.65, help="The path of test image"
    )
    parser.add_argument("--cpu", action="store_true", default=True, help="Run on cpu")

    opt = parser.parse_args()


    os.makedirs("./results_new", exist_ok=True)

    # 选择推理后端
    if opt.cpu:
        print("run on cpu...")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("run on gpu...")
            device = torch.device("cuda")
        else:
            print("run on cpu...")
            device = torch.device("cpu")

    # 解析yaml配置文件
    cfg = LoadYaml(opt.yaml)
    print(cfg)
    anchors = torch.tensor(cfg.anchor_sizes, device=device)

    val_dataset = TensorDataset(
        path="D:/data/dataset/val.txt",
        img_width=1280,
        img_height=720,
        aug=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
    )

    # 模型加载
    print("load weight from:%s" % opt.weight)
    model = Detector(cfg.category_num, True, anchor_num=len(cfg.anchor_sizes)).to(device)
    model.load_state_dict(torch.load(opt.weight, map_location=device))

    loss_fn = DetectorLoss(cfg.anchor_sizes, device)
    # model = torch.compile(model)
    # sets the module in eval node
    model.eval()
    for imgs, targets in tqdm(val_dataloader):
        # 数据预处理
        imgs: torch.Tensor = imgs.to(device).float() / 255.0
        targets = targets.to(device)
        #cv2.imshow("1", imgs[0].permute(1, 2, 0).numpy())
        # 模型推理
        res_imgs = torch.nn.functional.interpolate(imgs, (352, 352), mode='bilinear')
        #cv2.imshow("1", res_imgs[0].permute(1, 2, 0).numpy())
        start = time.perf_counter()
        # with torch.no_grad():
        preds = model(res_imgs)
        end = time.perf_counter()
        process_time = (end - start) * 1000.0
        print("forward time:%fms" % process_time)
        output = handle_preds(preds, anchors, device, opt.thresh)
        iou_loss, obj_loss, cls_loss, tot_loss = loss_fn(preds, targets)
        # print(iou_loss, obj_loss, cls_loss, tot_loss)
        # 加载label names
        LABEL_NAMES = []
        with open("D:/data/dataset/robocon.names", "r") as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        imgs = imgs[0].permute(1, 2, 0).numpy()
        H, W, _ = imgs.shape
        scale_h, scale_w = H / cfg.input_height, W / cfg.input_width
        imgs = np.ascontiguousarray(imgs) * 255
        # 绘制预测框
        for box in output[0]:
            # print(box)
            box = box.tolist()

            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)
            # print(imgs.shape, (x1, y1), (x2, y2))
            cv2.rectangle(imgs, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                imgs, "%.2f" % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2
            )
            cv2.putText(imgs, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        #cv2.imshow("1", imgs)
        cv2.imwrite("./results_new/" + f"result_{iou_loss.item():.6f}.png", imgs)
        '''
        ori_img = cv2.imread(os.path.join(opt.img_folder, file))
        res_img = cv2.resize(
            ori_img, (cfg.input_width, cfg.input_height), interpolation=cv2.INTER_LINEAR
        )
        img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

        # 模型推理
        start = time.perf_counter()
        # with torch.no_grad():
        preds = model(img)
        end = time.perf_counter()
        process_time = (end - start) * 1000.0
        print("forward time:%fms" % process_time)

        # 特征图后处理
        output = handle_preds(preds, anchors, device, opt.thresh)

        # 加载label names
        LABEL_NAMES = []
        with open(cfg.names, "r") as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())

        H, W, _ = ori_img.shape
        scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

        # 绘制预测框
        for box in output[0]:
            # print(box)
            box = box.tolist()

            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)

            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                ori_img, "%.2f" % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2
            )
            cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        cv2.imwrite("./results_new/" + f"result_{file}", ori_img)'''

# python train.py --yaml configs/robocon.yaml --weight checkpoint\weight_AP05_0.875423_200-epoch.pth --img dataset\val
