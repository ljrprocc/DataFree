import os
import shutil

if __name__ == '__main__':
    root = '/data/lijingru/timagenet/tiny-imagenet-200/'
    txt_file = 'val_annotations.txt'
    f = open(os.path.join(root, 'val/', txt_file), 'r').readlines()
    for l in f:
        l = l.strip()
        f_name, folder_name = l.split('\t')[0], l.split('\t')[1]
        folder = os.path.join(root, 'val_split/', folder_name)
        ori_f = os.path.join(root, 'val/images/', f_name)
        dst_f = os.path.join(folder, f_name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        shutil.copy(ori_f, dst_f)