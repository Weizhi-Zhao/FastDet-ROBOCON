import os
import cv2
import onnx
import time
import argparse
from onnxsim import simplify

import torch
from utils.tool import *
from module.detector import Detector

import pykinect_azure as pykinect

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="configs/robocon.yaml", help='.yaml config')
    parser.add_argument('--weight', type=str, default='checkpoint/weight_AP05_0.830133_210-epoch.pth',
                        help='.weight config')
    # parser.add_argument('--img', type=str, default='',
    #                     help='The path of test image')
    parser.add_argument('--thresh', type=float, default=0.65,
                        help='The path of test image')
    parser.add_argument('--onnx', action="store_true",
                        default=False, help='Export onnx file')
    parser.add_argument('--torchscript', action="store_true",
                        default=False, help='Export torchscript file')
    parser.add_argument('--cpu', action="store_true",
                        default=False, help='Run on cpu')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    assert os.path.exists(opt.weight), "请指定正确的模型路径"
    # assert os.path.exists(opt.img_folder), "请指定正确的测试图像路径"

    if not os.path.exists('./results'):
        os.makedirs('./results')

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

    # 模型加载
    print("load weight from:%s" % opt.weight)
    model = Detector(cfg.category_num, True).to(device)
    model.load_state_dict(torch.load(opt.weight, map_location=device))
    # sets the module in eval node
    model.eval()

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

	# Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
	# print(device_config)

	# Start device
    kinect_device = pykinect.start_device(config=device_config)

    cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)

    while True:
        # Get capture
        capture = kinect_device.update()

		# Get the color image from the capture
        ret, color_image = capture.get_color_image()

        if not ret:
            continue
        res_img = cv2.resize(
            color_image, (cfg.input_width, cfg.input_height), interpolation=cv2.INTER_LINEAR)
        img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

        # 模型推理
        start = time.perf_counter()
        preds = model(img)
        # 特征图后处理
        output = handle_preds(preds, device, opt.thresh)
        end = time.perf_counter()
        process_time = (end - start) * 1000.
        print("forward time:%fms" % process_time)
        # 加载label names
        LABEL_NAMES = []
        with open(cfg.names, 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        
        H, W, _ = color_image.shape
        scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

        # 绘制预测框
        for box in output[0]:
            print(box)
            box = box.tolist()

            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(color_image, '%.2f' % obj_score,
                        (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(color_image, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        # Plot the image
        cv2.imshow("Color Image",color_image)
		
		# Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break

# python test_img_folder.py --yaml configs/robocon.yaml --weight checkpoint\weight_AP05_0.830133_210-epoch.pth --img dataset\val   