# 按标签分文件夹存放不相关图片
import os
import shutil


def organize_images_by_label(root):
    source_directory = os.path.join(root, "util_pic")
    files = os.listdir(source_directory)
    for file in files:
        if file.endswith('.jpg'):
            filename, ext = os.path.splitext(file)  # 获取文件名和扩展名
            label = filename.rsplit('_', 1)[0]  # 获取label部分
            target_folder = os.path.join(root, label)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            source_file = os.path.join(source_directory, file)
            target_file = os.path.join(target_folder, file)

            shutil.copy(source_file, target_file)
            print(f"Copied {file} to {target_folder}")


root = './data/base'
organize_images_by_label(root)
