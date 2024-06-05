import os
import random
import shutil
from tqdm import tqdm

file_name_list = os.listdir('./dataset/images')
file_name_list = [name.split('.')[0] for name in file_name_list]
random.shuffle(file_name_list) # no return

dataset_len = len(file_name_list)
train_set_name = file_name_list[:round(dataset_len*0.8)]
val_set_name = file_name_list[round(dataset_len*0.8):dataset_len]

target_train_dir = './dataset/train'
target_val_dir = './dataset/val'

if not os.path.exists(target_train_dir):
    os.makedirs(target_train_dir)

if not os.path.exists(target_val_dir):
    os.makedirs(target_val_dir)

for name in tqdm(train_set_name):
    shutil.copyfile(os.path.join('./dataset/images', name + '.png'), os.path.join(target_train_dir, name + '.png'))
    if os.path.exists(os.path.join('./dataset/labels', name + '.txt')):
        shutil.copyfile(os.path.join('./dataset/labels', name + '.txt'), os.path.join(target_train_dir, name + '.txt'))
    else:
        with open(os.path.join(target_train_dir, name + '.txt'), 'w') as f:
            f.write('')

for name in tqdm(val_set_name):
    shutil.copyfile(os.path.join('./dataset/images', name + '.png'), os.path.join(target_val_dir, name + '.png'))
    if os.path.exists(os.path.join('./dataset/labels', name + '.txt')):
        shutil.copyfile(os.path.join('./dataset/labels', name + '.txt'), os.path.join(target_val_dir, name + '.txt'))
    else:
        with open(os.path.join(target_val_dir, name + '.txt'), 'w') as f:
            f.write('')
