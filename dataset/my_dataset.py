import os
import numpy as np
import torch.nn.functional as F
import glob
import json
import re
import random
import torch
import scipy.ndimage as ndimage
import SimpleITK as sitk
import nrrd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
lower = -250



class ToothDataset(Dataset):
    def __init__(self, root, mode="train"):
        # 初始化各参数
        self.root = root
        self.mode = mode
        # self.window_width = 1000
        # self.window_level = 500
        # 加载数据集文件列表
        self.image_dir_path = os.path.join(self.root, mode, "images")
        self.label_dir_path = os.path.join(self.root, mode, "labels")
        self.imgs = []
        for image_name in os.listdir(self.image_dir_path):
            image_path = os.path.join(self.image_dir_path, image_name)
            if os.path.isfile(image_path):
                self.imgs.append(image_name)
        # 读取索引文件
        json_file = './dataset/index_to_class.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.index_to_class_dict = json.load(json_file)
        self.class_to_index_dict = {}
        for key, val in self.index_to_class_dict.items():
            self.class_to_index_dict[val] = key
        json_file.close()



    def __getitem__(self, index):
        # 加载图像数据
        image_path = os.path.join(self.image_dir_path, self.imgs[index])
        image_array = np.load(image_path)
        # 加载标签图像
        label_path = os.path.join(self.label_dir_path, self.imgs[index])
        label_array = np.load(label_path)

        # clip
        image_array = self.clip(image_array)

        # 标准化
        image_array = self.standardize(image_array)

        # plt.hist(label_crop.flatten(), bins=50, range=(0, 50), color="g", histtype="bar", rwidth=1, alpha=0.6)
        # plt.show()

        # 维度变换
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0)  # [1, 160, 160, 64]
        image_tensor = image_tensor.permute(0, 3, 1, 2)  # [1, 64, 160, 160]

        label_tensor = torch.FloatTensor(label_array)  # [160, 160, 64]
        label_tensor = label_tensor.permute(2, 0, 1)

        return image_tensor, label_tensor



    def resample(self, image, label, spacing, newSpacing=[0.25, 0.25, 0.25]):
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



    def clip(self, image):
        """
        约束上下界
        Args:
            image: 图像数据

        Returns:

        """
        # 上下界数值
        upper_bound = 1669.7011767578128
        lower_bound = -3235.0
        # clip
        new_image = np.clip(image, lower_bound, upper_bound)
        return new_image



    def standardize(self, image_np):
        """
        标准化
        Args:
            image_np: 原图数据

        Returns:

        """
        # 均值
        mean = 84.87565
        # 标准差
        std =  665.2517
        # 标准化
        image_np = (image_np - mean) / std

        return image_np



    def randomCrop(self, image_np, label_np, fix_shape=[160, 160, 64]):
        """
        随机裁剪
        Args:
            image_np: 原图图像数据
            label_np: 标签图像数据
            fix_shape: 固定裁剪大小

        Returns:

        """
        # 获取数据维度
        ori_shape = label_np.shape
        # 初始化裁剪数组
        image_crop = np.zeros(fix_shape, dtype=image_np.dtype)
        label_crop = np.zeros(fix_shape, dtype=label_np.dtype)
        # 找出前景区域
        x_min, x_max, y_min, y_max, z_min, z_max = 0, ori_shape[0] - 1, 0, ori_shape[1] - 1, 0, ori_shape[2] - 1
        # x, y, z = label_np.nonzero()
        # x_min, x_max, y_min, y_max, z_min, z_max = x.min(), x.max(), y.min(), y.max(), z.min(), z.max()
        # x_min = max(0, x_min - 16)
        # x_max = min(ori_shape[0] - 1, x_max + 16)
        # y_min = max(0, y_min - 16)
        # y_max = min(ori_shape[1] - 1, y_max + 16)
        # z_min = max(0, z_min - 16)
        # z_max = min(ori_shape[2] - 1, z_max + 16)
        # 随机获取裁剪区间
        xs = random.randint(x_min, max(x_min, x_max - fix_shape[0] + 1))
        xe = min(xs + fix_shape[0], x_max + 1)
        ys = random.randint(y_min, max(y_min, y_max - fix_shape[1] + 1))
        ye = min(ys + fix_shape[1], y_max + 1)
        zs = random.randint(z_min, max(z_min, z_max - fix_shape[2] + 1))
        ze = min(zs + fix_shape[2], z_max + 1)
        resize_shape = [xe - xs, ye - ys, ze - zs]
        # print(resize_shape)
        # 裁剪
        image_crop[:resize_shape[0], :resize_shape[1], :resize_shape[2]] = image_np[xs:xe, ys:ye, zs:ze]
        label_crop[:resize_shape[0], :resize_shape[1], :resize_shape[2]] = label_np[xs:xe, ys:ye, zs:ze]

        return image_crop, label_crop, resize_shape



    def load_label_nrrd(self, path):
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
        segment_dict = self.class_to_index_dict.copy()
        for key in segment_dict.keys():
            segment_dict[key] = {"index": int(segment_dict[key]), "color": None, "labelValue": None}
        # 整理每个种类标签的信息
        classes = ["ul1", "ul2", "ul3", "ul4", "ul5", "ul6", "ul7", "ul8",
                   "ur1", "ur2", "ur3", "ur4", "ur5", "ur6", "ur7", "ur8",
                   "bl1", "bl2", "bl3", "bl4", "bl5", "bl6", "bl7", "bl8",
                   "br1", "br2", "br3", "br4", "br5", "br6", "br7", "br8",
                   "gum", "implant"]
        for key, val in options.items():
            searchObj = re.search(r'^Segment(\d+)_Name$', key)
            if searchObj is not None:
                segment_id = searchObj.group(1)
                # 获取颜色
                segment_color_key = "Segment" + str(segment_id) + "_Color"
                color = options.get(segment_color_key, None)
                if color is not None:
                    tmpColor = color.split()
                    color = [int(255*float(c)) for c in tmpColor]
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
                data[data==val["labelValue"]] = -val["index"]

        data = -data
        # for i in range(0, 35):
        #     print(str(i), (data==i).sum())
        # print(options)

        # plt.hist(data.flatten(), bins=50, range=(1, 50), color="g", histtype="bar", rwidth=1, alpha=0.6)
        # plt.show()

        spacing = [v[i] for i, v in enumerate(options["space directions"])]

        return data, segment_dict, spacing



    def resieze(self, image, label, new_shape=[192, 192, 72]):
        """
            缩放图像并统一大小
        Args:
            image: 原图
            label: 标注
            new_shape: 大小

        Returns:

        """
        # 初始化张量
        ori_shape = image.shape
        new_image = np.zeros(new_shape)
        new_label = np.zeros(new_shape)
        # 确定缩放比例
        shape_ratio = np.array(new_shape) / np.array(ori_shape)
        min_ratio = shape_ratio.min()
        resize_shape = [int(float(min_ratio)*shape)for shape in ori_shape]
        # 缩放原图
        tmp_image = torch.FloatTensor(image).unsqueeze(dim=0).unsqueeze(dim=0)
        tmp_image = F.interpolate(tmp_image, size=resize_shape, mode='trilinear',
                                  align_corners=True).squeeze().detach().numpy()
        # 缩放标注图像
        tmp_label = torch.FloatTensor(label).unsqueeze(dim=0).unsqueeze(dim=0)
        tmp_label = F.interpolate(tmp_label, size=resize_shape, mode='nearest',).squeeze().detach().numpy()
        # 填充
        new_image[:resize_shape[0], :resize_shape[1], :resize_shape[2]] = tmp_image[:, :, :]
        new_label[:resize_shape[0], :resize_shape[1], :resize_shape[2]] = tmp_label[:, :, :]

        # plt.hist(new_label.flatten(), bins=50, range=(1, 50), color="g", histtype="bar", rwidth=1, alpha=0.6)
        # plt.show()

        return new_image, new_label, resize_shape, ori_shape



    def __len__(self):
        return len(self.imgs)



    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)  # transposed
        return torch.stack(images, 0), torch.stack(labels, 0)









class CTDataLoader(Dataset):
    def __init__(self, root = '/data/weihao/pre-KiTS-3mm/',mode = 'train', slice_number = None, scale=False, rotate=False, flip = False, glob_flag=False, use_weight=False):
        '''
        :param root:
        :param mode:
        '''
        super(CTDataLoader, self).__init__()
        self.root = root
        self.mode = mode
        self.scale = scale # the True of False
        self.slice_number = slice_number
        self.rotate = rotate # the True of False
        self.flip = flip # the True of False
        # self.num_class = config.num_class # the class number which include the background class
        self.glob_flag = glob_flag
        if self.mode == 'train':
            # self.image_dir = os.path.join(self.root, 'train/CT/')
            # self.label_dir = os.path.join(self.root, 'train/GT/')
            self.image_dir = os.path.join(self.root, 'train_original.txt') # read the case list by a txt file
            self.hard_image_dir = os.path.join(self.root, 'train_selected.txt')

            hf = open(self.hard_image_dir, 'r')
            self.image_hard_path_list = hf.readlines()
            self.image_hard_path_list = [file[:-1] for file in self.image_hard_path_list]
            self.image_hard_path_list.sort()

        elif self.mode == 'val':
            # self.image_dir = os.path.join(self.root, 'val/CT/')
            # self.label_dir = os.path.join(self.root, 'val/GT/')
            self.image_dir = os.path.join(self.root, 'val_original.txt')  # read the case list by a txt file

        # self.image_path_list = glob.glob(self.image_dir+'*.nii')
        # self.image_path_list = [file for file in self.image_path_list if 'back' not in file]
        # self.image_path_list.sort()

        f = open(self.image_dir, 'r')
        self.image_path_list = f.readlines()
        self.image_path_list = [file[:-1] for file in self.image_path_list]
        self.image_path_list.sort()

        from tqdm import tqdm
        if use_weight:
            labelweights = np.zeros(3)
            for image_path in tqdm(self.image_path_list,total=len(self.image_path_list)):
                label_path = image_path.replace('CT', 'GT')
                label_path = label_path.replace('img', 'label')
                label = sitk.ReadImage(label_path, sitk.sitkUInt8)
                label_array = sitk.GetArrayFromImage(label)  # [384, 240, 80] ndarray in 0,1,2
                tmp, _ = np.histogram(label_array, range(4))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
            print(self.labelweights)
        else:
            self.labelweights = np.ones(3)

    def __getitem__(self, index):
        """
        :param index:
        :return: torch.Size([batch, 1, 48, 384, 240]) torch.Size([batch, 48, 384, 240])
        """
        # if self.mode == 'train' and random.uniform(0,1) <= 0.7:
        #     image_path = self.image_hard_path_list[index%50] #只取前50难的case
        # else:
        #     image_path = self.image_path_list[index]
        image_path = self.image_path_list[index]
        label_path = image_path.replace('CT','GT')
        label_path = label_path.replace('img','label')

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        label = sitk.ReadImage(label_path, sitk.sitkUInt8)

        label_array = sitk.GetArrayFromImage(label) # [384, 240, 80] ndarray in 0,1,2
        #label_array[label_array != 2] = 0 #Get the Tumor label

        image_array = sitk.GetArrayFromImage(image) # [384, 240, 80] ndarray in range [-250, 250]
        image_array = self.clip_intensity(image_array)

        cube_glob = image_array
        label_glob = label_array
        # if config.num_class == 2 and config.organ_type[0] == 'tumor':
        #     label_array[label_array < 2] = 0
        #     label_array[label_array > 0] = 1
        # elif config.num_class ==2 and config.organ_type[0] == 'kidney':
        #     label_array[label_array > 0] = 1
        #
        # assert len(label_array.shape) == 3,'the error in crop'

        # if self.num_class is not None: # check the correctness of label
        #     assert label_array.min() >= 0 and label_array.max() < self.num_class, \
        #         'the range of file {} should be [0,num_class-1], but min {}--max{}'.format(image_path, label_array.min(),label_array.max())
        _, _, _, _, z_min, z_max = self.getBoundbox(label_glob)
        if self.slice_number is not None:
            # sample the tensor only in axis Z, output (x,y,slice_number)
            start_slice = random.randint(max(z_min - 16,0), min(max(z_max-8, z_min), label_glob.shape[2]-self.slice_number))#(0, image_array.shape[-1] -self.slice_number)
            end_slice = start_slice + self.slice_number - 1
            # print(start_slice, end_slice, label_glob.shape[2])
            if end_slice >= label_glob.shape[2]:
                print('no!!!!')
                exit()
            image_array = image_array[:,:, start_slice:end_slice + 1]
            label_array = label_array[:,:, start_slice:end_slice + 1]
        else:
            start_slice = 0
            end_slice = 0

        # array to tensor

        image_array = torch.FloatTensor(image_array).unsqueeze(0) # [1, 384, 240, 80]
        image_array = image_array / 250.0 # rescale the range of intensity
        image_array = image_array.permute(0,3,1,2) # [1, 80, 384, 240]
        label_array = torch.FloatTensor(label_array) # [384, 240, 80]
        label_array = label_array.permute(2,0,1) # [80, 384, 240] nn.CrossEntropyLoss()

        cube_glob = torch.FloatTensor(cube_glob).unsqueeze(0) # [1, 384, 240, 80]
        cube_glob = cube_glob / 250.0 # rescale the range of intensity
        cube_glob = cube_glob.permute(0,3,1,2) # [1, 80, 384, 240]
        label_glob = torch.FloatTensor(label_glob) # [384, 240, 80]
        label_glob = label_glob.permute(2,0,1) # [80, 384, 240] nn.CrossEntropyLoss()

        #assert label_array.max() == 2, 'label needs to have 2 labels'
        # if self.num_class is not None: # check the correctness of label
        #     assert label_array.min() >= 0 and label_array.max() < self.num_class, \
        #     'the range of file {} should be [0,num_class-1], but min {}--max{}'.format(image_path, label_array.min(),label_array.max())
        if self.glob_flag:
            return image_array, label_array, cube_glob, start_slice, end_slice, label_glob
        else:
            return image_array, label_array

    def __len__(self):
        return len(self.image_path_list)

    def clip_intensity(self, ct_array, intensity_range=(-250,250)):
        ct_array[ct_array>intensity_range[1]] = intensity_range[1]
        ct_array[ct_array<intensity_range[0]] = intensity_range[0]
        return ct_array

    def zoom(self, ct_array, seg_array, patch_size):

        shape = ct_array.shape # [384, 240, 80]
        length_hight = int(shape[0] * patch_size)
        length_width = int(shape[1] * patch_size)

        length = int(256 * patch_size)

        x1 = int(random.uniform(0, shape[0] - length_hight))
        y1 = int(random.uniform(0, shape[1] - length_width))

        x2 = x1 + length_hight
        y2 = y1 + length_width

        ct_array = ct_array[x1:x2 + 1, y1:y2 + 1,:]
        seg_array = seg_array[x1:x2 + 1, y1:y2 + 1,:]

        with torch.no_grad():

            ct_array = torch.FloatTensor(ct_array).unsqueeze(dim=0).unsqueeze(dim=0)
            ct_array = ct_array
            ct_array = F.interpolate(ct_array, (shape[0], shape[1], shape[2]), mode='trilinear', align_corners=True).squeeze().detach().numpy()

            seg_array = torch.FloatTensor(seg_array).unsqueeze(dim=0).unsqueeze(dim=0)
            seg_array = seg_array
            seg_array = F.interpolate(seg_array, (shape[0], shape[1], shape[2])).squeeze().detach().numpy()

            return ct_array, seg_array

    def randomCrop(self, input_image, input_label, crop_size=(96,96,32)):
        '''
        random crop the cubic in object region
        :param input_image:
        :param crop_size:
        :return:
        '''
        assert input_label.shape == input_image.shape,'the shape of mask and input_image should be same'
        assert isinstance(input_image,np.ndarray),'the input_image should be np.ndarray'

        # randm crop the cubic
        new_x = random.randint(0, input_image.shape[0] - crop_size[0])
        end_x = new_x + crop_size[0] - 1
        new_y = random.randint(0, input_image.shape[1] - crop_size[1])
        end_y = new_y + crop_size[1] - 1
        new_z = random.randint(0, input_image.shape[2] - crop_size[2])
        end_z = new_z + crop_size[2] - 1
        #
        image_array = input_image[new_x:end_x+1, new_y:end_y+1, new_z:end_z]
        label_array = input_label[new_x:end_x+1, new_y:end_y+1, new_z:end_z]

        return image_array, label_array

    def resample(self, input_image, target_spacing = (0.5, 0.5, 0.5)):
        '''
        resample the CT image
        :parm input_image:
        :param target_spacing: 
        '''
        assert isinstance(input_image, sitk.Image), 'the input_image should be the object of SimpleITK.SimpleITK.Image'
        origin_spacing = input_image.GetSpacing()
        origin_size = input_image.GetSize()
        scale = [target_spacing[index]/origin_spacing[index] for index in range(len(origin_size))]
        new_size = [int(origin_size[index]/scale[index]) for index in range(len(origin_size))]
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(0)
        resample.SetOutputSpacing(target_spacing)
        resample.SetOutputOrigin(input_image.GetOrigin())
        resample.SetOutputDirection(input_image.GetDirection())
        resample.SetSize(new_size)
        new_image = resample.Execute(input_image)
        return new_image
        
    def getBoundbox(self, input_array):
        '''
        get the bouding box for input_array (the non-zero range is our object)
        '''
        assert isinstance(input_array, np.ndarray)
        x,y,z = input_array.nonzero()
        return [x.min(),x.max(), y.min(), y.max(), z.min(),z.max()]

# the test code
if __name__ == '__main__':

    data_train = CTDataLoader(mode='train',use_weight=True)
    test_image, test_label = data_train.__getitem__(0)
    numbers = data_train.__len__()
    print(test_image.shape)
    print(test_label.shape)
    print('the number of cases in dataset: ', data_train.__len__())
    
