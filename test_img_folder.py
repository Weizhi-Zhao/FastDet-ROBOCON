import os
import cv2
import onnx
import time
import argparse
from onnxsim import simplify

import torch
from utils.tool import *
from module.detector import Detector

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
    parser.add_argument(
        "--onnx", action="store_true", default=False, help="Export onnx file"
    )
    parser.add_argument(
        "--torchscript",
        action="store_true",
        default=False,
        help="Export torchscript file",
    )
    parser.add_argument("--cpu", action="store_true", default=True, help="Run on cpu")

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    assert os.path.exists(opt.weight), "请指定正确的模型路径"
    assert os.path.exists(opt.img_folder), "请指定正确的测试图像路径"

    # TODO
    if not os.path.exists("./results"):
        os.makedirs("./results")

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

    # 模型加载
    print("load weight from:%s" % opt.weight)
    model = Detector(cfg.category_num, True, anchor_num=len(cfg.anchor_sizes)).to(device)
    model.load_state_dict(torch.load(opt.weight, map_location=device))
    # model = torch.compile(model)
    # sets the module in eval node
    model.eval()
    for file in os.listdir(opt.img_folder):
        if not file.endswith(".png"):
            continue

        # 数据预处理
        ori_img = cv2.imread(os.path.join(opt.img_folder, file))
        res_img = cv2.resize(
            ori_img, (cfg.input_width, cfg.input_height), interpolation=cv2.INTER_LINEAR
        )
        img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

        # 导出onnx模型
        if opt.onnx:
            torch.onnx.export(
                model,  # model being run
                # model input (or a tuple for multiple inputs)
                img,
                # where to save the model (can be a file or file-like object)
                "./FastestDet.onnx",
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=11,  # the ONNX version to export the model to
                do_constant_folding=True,
            )  # whether to execute constant folding for optimization
            # onnx-sim
            onnx_model = onnx.load("./FastestDet.onnx")  # load onnx model
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            print("onnx sim sucess...")
            onnx.save(model_simp, "./FastestDet.onnx")

        # 导出torchscript模型
        if opt.torchscript:
            import copy

            model_cpu = copy.deepcopy(model).cpu()
            x = torch.rand(1, 3, cfg.input_height, cfg.input_width)
            mod = torch.jit.trace(model_cpu, x)
            mod.save("./FastestDet.pt")
            print(
                "to convert torchscript to pnnx/ncnn: ./pnnx FastestDet.pt inputshape=[1,3,%d,%d]"
                % (cfg.input_height, cfg.input_height)
            )

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

        cv2.imwrite("./results_new/" + f"result_{file}", ori_img)

# python test_img_folder.py --yaml configs/robocon.yaml --weight checkpoint\weight_AP05_0.875423_200-epoch.pth --img dataset\val
