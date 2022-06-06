import shutil

import nrrd
from nibabel.viewers import OrthoSlicer3D
from matplotlib import pyplot as plt
import SimpleITK as sitk
import collections
import numpy as np
import re
import os
import torch
import torch.nn.functional as F
import json
from functools import reduce



def load_nrrd_file(path):
    """
    读取nrrd文件
    Args:
        path: 文件路径

    Returns:

    """
    data, options = nrrd.read(path)  # 读入 nrrd 文件
    # print(data.ndim)
    if data.ndim == 4:
        data = data[0, :, :, :]
    x, y, z = data.nonzero()
    # print(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    # print(data)
    # print(data.shape)
    # plt.hist(data[0,:,:,:].flatten(), bins=data[0,:,:,:].max(), color="g", histtype="bar", rwidth=1, alpha=0.6)
    # plt.show()
    # OrthoSlicer3D(data[0,:,:,:]).show()
    # print(type(data))
    # print(options)

    # 初始化标记字典
    segment_dict = {}
    directions = ["ul", "ur", "bl", "br"]
    for i in range(1, 9):
        for d in directions:
            key = d + str(i)
            segment_dict[key] = {"color": None, "extent": None, "labelValue": None}
    # 整理每个种类标签的信息
    for key in segment_dict.keys():
        for i in range(50):
            segment_name_key = "Segment" + str(i) + "_Name"
            segment_name_val = options.get(segment_name_key, None)
            if segment_name_val is None:
                break
            if segment_name_val == key:
                color_key = "Segment" + str(i) + "_Color"
                segment_dict[key]["color"] = options.get(color_key, None)
                extent_key = "Segment" + str(i) + "_Extent"
                segment_dict[key]["extent"] = options.get(extent_key, None)
                label_value_key = "Segment" + str(i) + "_LabelValue"
                segment_dict[key]["labelValue"] = options.get(label_value_key, None)
                # print("{:<7d}".format(np.sum(data==int(segment_dict[key]["labelValue"]))), end="")
                break
    # print()

    segment_dict = collections.OrderedDict(sorted(segment_dict.items(), key=lambda t: t[0]))
    return segment_dict



def compare_label_nrrd():
    dict1 = load_nrrd_file(r"./data/train/labels/1_1.nrrd")
    dict2 = load_nrrd_file(r"./data/train/labels/1_2.nrrd")
    dict3 = load_nrrd_file(r"./data/train/labels/2_0.nrrd")
    dict4 = load_nrrd_file(r"./data/train/labels/7_0.nrrd")
    dict5 = load_nrrd_file(r"./data/train/labels/12_2.nrrd")
    dict6 = load_nrrd_file(r"./data/train/labels/15_2.nrrd")
    dict7 = load_nrrd_file(r"./data/train/labels/19_0.nrrd")
    dict8 = load_nrrd_file(r"./data/train/labels/21_0.nrrd")
    dict9 = load_nrrd_file(r"./data/val/labels/1_0.nrrd")
    dict10 = load_nrrd_file(r"./data/val/labels/19_2.nrrd")
    dicts = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10]

    directions = ["ul", "ur", "bl", "br"]
    for d in directions:
        for i in range(1, 9):
            key = d + str(i)
            color_arr = []
            label_value_arr = []
            for dict in dicts:
                if dict[key]["color"] is not None:
                    color_arr.append(dict[key]["color"])
                if dict[key]["labelValue"] is not None:
                    label_value_arr.append(dict[key]["labelValue"])
            color_flag = True
            label_value_flag = True
            if len(color_arr) > 0 and len(color_arr) != color_arr.count(color_arr[0]):
                color_flag = False
            if len(label_value_arr) > 0 and len(label_value_arr) != label_value_arr.count(label_value_arr[0]):
                label_value_flag = False
            print("{}: color:{}, label_value:{}".format(key, color_flag, label_value_flag))


def load_label_nrrd(path):
    # print(path)
    """
    读取nrrd文件
    Args:
        path: 文件路径

    Returns:

    """
    # 读入 nrrd 文件
    data, options = nrrd.read(path)
    if data.ndim == 4:
        data = data[0, :, :, :]
    # plt.hist(data.flatten(), bins=50, range=(1, 50), color="g", histtype="bar", rwidth=1, alpha=0.6)
    # plt.show()
    # 初始化标记字典
    # 读取索引文件
    json_file = './dataset/index_to_class.json'
    assert os.path.exists(json_file), "{} file not exist.".format(json_file)
    json_file = open(json_file, 'r')
    index_to_class_dict = json.load(json_file)
    class_to_index_dict = {}
    for key, val in index_to_class_dict.items():
        class_to_index_dict[val] = key
    json_file.close()
    segment_dict = class_to_index_dict.copy()
    for key in segment_dict.keys():
        segment_dict[key] = {"index": int(segment_dict[key]), "color": None, "labelValue": None}

    for key, val in options.items():
        searchObj = re.search(r'^Segment(\d+)_Name$', key)
        if searchObj is not None:
            segment_id = searchObj.group(1)
            # 获取颜色
            segment_color_key = "Segment" + str(segment_id) + "_Color"
            color = options.get(segment_color_key, None)
            if color is not None:
                tmpColor = color.split()
                color = [int(255 * float(c)) for c in tmpColor]
            segment_dict[val]["color"] = color
            # 获取标签值
            segment_label_value_key = "Segment" + str(segment_id) + "_LabelValue"
            labelValue = options.get(segment_label_value_key, None)
            if labelValue is not None:
                labelValue = int(labelValue)
            segment_dict[val]["labelValue"] = labelValue
    # 替换标签值
    for key, val in segment_dict.items():
        if val["labelValue"] is not None:
            # print(key, val["labelValue"])
            data[data == val["labelValue"]] = -val["index"]

    data = -data
    # for i in range(0, 35):
    #     print(str(i), (data==i).sum())
    # print(options)

    # plt.hist(data.flatten(), bins=50, range=(1, 50), color="g", histtype="bar", rwidth=1, alpha=0.6)
    # plt.show()

    spacing = [v[i] for i, v in enumerate(options["space directions"])]

    return data, segment_dict, spacing



def test_label_nrrds(dir_path) :
    for nrrd_name in os.listdir(dir_path):
        nrrd_path = os.path.join(dir_path, nrrd_name)
        print(nrrd_name + " : ")
        _, segment_dict, _ = load_label_nrrd(nrrd_path)
        cnt = 0
        label_value_set = set([])
        for key, val in segment_dict.items():
            if val["labelValue"] is not None:
                cnt += 1
                label_value_set.add(val["labelValue"])
        print(list(label_value_set), cnt, len(label_value_set))
        print()
        # print("{} : cnt: {}, set: {}".format(nrrd_name, cnt, len(label_value_set)))



def load_image_nrrd(dir_path):
    for nrrd_name in os.listdir(dir_path):
        nrrd_path = os.path.join(dir_path, nrrd_name)
        image = sitk.ReadImage(nrrd_path)
        size = image.GetSize()
        print("Image size:", size)
        spacing = image.GetSpacing()
        print("Image spacing:", spacing)
        direction = image.GetDirection()
        print("Image direction:", direction)
        origin = image.GetOrigin()
        print("Image origin:", origin)

        image_np = sitk.GetArrayFromImage(image)
        print(image_np.shape)
        print(np.unique(image_np), len(np.unique(image_np)))

        # plt.hist(image_np.flatten(), bins=50, color="g", histtype="bar", rwidth=1, alpha=0.6)
        # plt.show()
        print()



def calMeanAndStd(dir_path):
    voxels_all = []
    ave_x = []
    ave_y = []
    ave_z = []
    for nrrd_name in os.listdir(dir_path):
        # 读取原图图像数据
        image_path = os.path.join(dir_path, nrrd_name)
        image_np, _ = nrrd.read(image_path)
        # 读取标注图像数据
        label_path = image_path.replace("images", "labels")
        label_np, _, spacing = load_label_nrrd(label_path)
        print(spacing)
        # 重采样resample
        image_np, label_np = resample(image_np, label_np, spacing, newSpacing=[0.25, 0.25, 0.25])
        # 获取前景的百分之十
        print(reduce(lambda x, y: x*y, label_np.shape))
        x, y, z = label_np.nonzero()
        x_min, x_max, y_min, y_max, z_min, z_max = x.min(), x.max(), y.min(), y.max(), z.min(), z.max()
        ave_x.append(x_max - x_min + 1)
        ave_y.append(y_max - y_min + 1)
        ave_z.append(z_max - z_min + 1)
        print(x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)
        print(reduce(lambda x, y: x * y, image_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1].shape))
        voxels = list(image_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1].flatten()[::10])
        print()
        # 添加到总体列表中
        voxels_all.extend(voxels)
    # 转成ndarray格式
    voxels_all = np.array(voxels_all)
    ave_x_np = np.array(ave_x)
    ave_y_np = np.array(ave_y)
    ave_z_np = np.array(ave_z)
    # 计算均值
    mean = np.mean(voxels_all)
    # 计算标准差
    std = np.std(voxels_all)
    # 获得99.5%分位数
    percentile_99_5 = np.percentile(voxels_all, 99.5)
    # 获得0.5%分位数
    percentile_00_5 = np.percentile(voxels_all, 00.5)
    print("Mean: ", mean)
    print("Std: ", std)
    print("percentile_99.5%: ", percentile_99_5)
    print("percentile_0.5%: ", percentile_00_5)
    print("x_shape: ", int(ave_x_np.mean()))
    print("y_shape: ", int(ave_y_np.mean()))
    print("z_shape: ", int(ave_z_np.mean()))



def resample(image, label, spacing, newSpacing=[0.25, 0.25, 0.25]):
    """
    重采样
    Args:
        image: 原图图像
        label: 标签图像
        spacing: 体素间距
        newSpacing: 新体素间距

    Returns:

    """
    # 获得原尺寸
    ori_shapes = label.shape
    # 计算新尺寸
    newSize = [int(ori_shape * spacing[i] / newSpacing[i]) for i, ori_shape in enumerate(ori_shapes)]
    # 对原图图像进行线性插值
    tmp_image = torch.FloatTensor(image).unsqueeze(dim=0).unsqueeze(dim=0)
    tmp_image = F.interpolate(tmp_image, size=newSize, mode='trilinear',
                              align_corners=True).squeeze().detach().numpy()
    # 对标签图像进行最邻近插值
    tmp_label = torch.FloatTensor(label).unsqueeze(dim=0).unsqueeze(dim=0)
    tmp_label = F.interpolate(tmp_label, size=newSize, mode='nearest', ).squeeze().detach().numpy()

    return tmp_image, tmp_label



def preprocess_dataset(dataset_dir):

    # 初始化统一尺寸
    x_shape = 192
    y_shape = 192
    z_shape = 96

    # 创建目录结构
    path_tuple = os.path.split(dataset_dir)
    process_dataset_dir = os.path.join(path_tuple[0], path_tuple[1] + "_processed")
    if os.path.exists(process_dataset_dir):
        shutil.rmtree(process_dataset_dir)
    os.mkdir(process_dataset_dir)
    for sub_dir in ["train", "val"]:
        os.mkdir(os.path.join(process_dataset_dir, sub_dir))
        for subsub_dir in ["images", "labels"]:
            os.mkdir(os.path.join(process_dataset_dir, sub_dir, subsub_dir))

    for sub_dir in ["train", "val"]:
        old_image_dir = os.path.join(dataset_dir, sub_dir, "images")
        old_label_dir = os.path.join(dataset_dir, sub_dir, "labels")
        new_image_dir = os.path.join(process_dataset_dir, sub_dir, "images")
        new_label_dir = os.path.join(process_dataset_dir, sub_dir, "labels")
        for image_name in os.listdir(old_image_dir):
            name_split = image_name.split(".")
            old_image_path = os.path.join(old_image_dir, image_name)
            old_label_path = os.path.join(old_label_dir, image_name)
            new_image_path = os.path.join(new_image_dir, name_split[0] + ".npy")
            new_label_path = os.path.join(new_label_dir, name_split[0] + ".npy")

            # 读取原图图像数据
            image_np, _ = nrrd.read(old_image_path)
            image_np = image_np.astype(np.float32)
            # 读取标注图像数据
            label_np, _, spacing = load_label_nrrd(old_label_path)
            label_np = label_np.astype(np.uint8)
            # 重采样resample
            image_np, label_np = resample(image_np, label_np, spacing, newSpacing=[0.25, 0.25, 0.25])
            # 获取roi
            x, y, z = label_np.nonzero()
            x_min, x_max, y_min, y_max, z_min, z_max = x.min(), x.max(), y.min(), y.max(), z.min(), z.max()
            # 剪切
            image_np = image_np[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
            label_np = label_np[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
            # 对原图图像进行线性插值
            tmp_image = torch.FloatTensor(image_np).unsqueeze(dim=0).unsqueeze(dim=0)
            image_np = F.interpolate(tmp_image, size=[x_shape, y_shape, z_shape], mode='trilinear',
                                      align_corners=True).squeeze().detach().numpy()
            # 对标签图像进行最邻近插值
            tmp_label = torch.FloatTensor(label_np).unsqueeze(dim=0).unsqueeze(dim=0)
            label_np = F.interpolate(tmp_label, size=[x_shape, y_shape, z_shape], mode='nearest', ).squeeze().detach().numpy()
            # 保存
            np.save(new_image_path, image_np)
            np.save(new_label_path, label_np)




def calNpyMeanAndStd(dir_path):
    voxels_all = []
    for npy_name in os.listdir(dir_path):
        # 读取原图图像数据
        image_path = os.path.join(dir_path, npy_name)
        image_np = np.load(image_path)
        # 读取标注图像数据
        label_path = image_path.replace("images", "labels")
        label_np = np.load(label_path)
        # 获取前景的百分之十
        print(reduce(lambda x, y: x*y, label_np.shape))
        x, y, z = label_np.nonzero()
        x_min, x_max, y_min, y_max, z_min, z_max = x.min(), x.max(), y.min(), y.max(), z.min(), z.max()
        print(x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)
        print(reduce(lambda x, y: x * y, image_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1].shape))
        voxels = list(image_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1].flatten()[::10])
        print()
        # 添加到总体列表中
        voxels_all.extend(voxels)
    # 转成ndarray格式
    voxels_all = np.array(voxels_all)
    # 计算均值
    mean = np.mean(voxels_all)
    # 计算标准差
    std = np.std(voxels_all)
    # 获得99.5%分位数
    percentile_99_5 = np.percentile(voxels_all, 99.5)
    # 获得0.5%分位数
    percentile_00_5 = np.percentile(voxels_all, 00.5)
    print("Mean: ", mean)
    print("Std: ", std)
    print("percentile_99.5%: ", percentile_99_5)
    print("percentile_0.5%: ", percentile_00_5)




if __name__ == '__main__':
    # 比较标签文件中同Name的Segment的颜色是否相同
    # compare_label_nrrd()

    # 测试标签文件中的labelValue个数和实际数据的类别个数
    # test_label_nrrds(r"./data/src_10/train/labels")

    # 查看原图像文件的信息
    # load_image_nrrd(r"./data/src_10/train/images")

    # 计算数据集的总均值和标准差
    # calMeanAndStd(r"./data/src_10/train/images")

    # 预处理数据集
    preprocess_dataset(r"./data/src_10")

    # 计算预处理后的数据集的总均值和标准差
    # calNpyMeanAndStd(r"./data/src_10_processed/train/images")