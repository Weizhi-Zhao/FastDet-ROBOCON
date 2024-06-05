import os

TRAIN_DIR = './dataset/train'
VAL_DIR = './dataset/val'

abs_train_img_path = [os.path.abspath(os.path.join(TRAIN_DIR, img_name)) for img_name in os.listdir(TRAIN_DIR)]
abs_train_img_path = [path + '\n' for path in abs_train_img_path if path.endswith('.png')]
abs_val_img_path = [os.path.abspath(os.path.join(VAL_DIR, img_name)) for img_name in os.listdir(VAL_DIR)]
abs_val_img_path = [path + '\n' for path in abs_val_img_path if path.endswith('.png')]


with open('./dataset/train.txt', 'w') as f:
    f.writelines(abs_train_img_path)

with open('./dataset/val.txt', 'w') as f:
    f.writelines(abs_val_img_path)