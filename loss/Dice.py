import torch
import torch.nn as nn
from config import num_class
import torch.nn.functional as F




def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return 1 - loss.sum() / N



class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation='sigmoid'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation

    def forward(self, pred, target):
        shape = target.shape
        organ_target = torch.zeros((target.size(0), num_class, shape[-3], shape[-2], shape[-1]))

        for organ_index in range(num_class):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :, :] = temp_target
            # organ_target: (B, 14, 48, 128, 128)

        organ_target = organ_target.cuda()

        total_loss = 0
        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(num_class):
            dice_loss = diceCoeffv2(pred[:, i], organ_target[:, i], activation=self.activation)
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / num_class



class BinaryDice(nn.Module):
    def __init__(self):
        super(BinaryDice, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size(0)
        # 平滑变量
        smooth = 1
        eps = 1e-5
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        return 1 - N_dice_eff.sum() / N


class MultiClassDiceLoss(nn.Module):
    def __init__(self,):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, input, target):
        """
            input tesor of shape = (N, C, H, W)
            target tensor of shape = (N, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        shape = target.shape
        organ_target = torch.zeros((target.size(0), num_class, shape[-3], shape[-2], shape[-1]))

        for organ_index in range(num_class):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :, :] = temp_target
            # organ_target: (B, 14, 48, 128, 128)

        organ_target = organ_target

        assert input.shape == organ_target.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDice()
        total_loss = 0

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(2, num_class):
            dice_loss = binaryDiceLoss(input[:, i], organ_target[:, i])
            total_loss += 10 * dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / num_class



def get_dice(pred, organ_target):
    # the organ_target should be one-hot code
    assert len(pred.shape) == len(organ_target.shape), 'the organ_target should be one-hot code'
    dice = 0
    for organ_index in range(num_class):
        P = pred[:, organ_index, :, :, :]
        _P = 1 - pred[:, organ_index, :, :, :]
        G = organ_target[:, organ_index, :, :, :]
        _G = 1 - organ_target[:, organ_index, :, :, :]
        mulPG = (P * G).sum(dim=1).sum(dim=1).sum(dim=1)
        mul_PG = (_P * G).sum(dim=1).sum(dim=1).sum(dim=1)
        mulP_G = (P * _G).sum(dim=1).sum(dim=1).sum(dim=1)

        dice += (mulPG + 1) / (mulPG + 0.8 * mul_PG + 0.2 * mulP_G + 1)
    # print(dice)
    # print(dice.size())
    return dice



class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        shape = target.shape
        organ_target = torch.zeros((target.size(0), num_class, shape[-3], shape[-2], shape[-1]))

        for organ_index in range(num_class):

            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :, :] = temp_target
            # organ_target: (B, 14, 48, 128, 128)

        organ_target = organ_target.cuda()

        return 1-get_dice(pred, organ_target).mean()




class DiceLoss_Focal(nn.Module):
    def __init__(self, has_softmax=True):
        self.has_softmax = has_softmax
        super().__init__()

    def forward(self, pred, target):
        if self.has_softmax:
            pred = F.softmax(pred, dim=1)
        shape = target.shape
        organ_target = torch.zeros((target.size(0), num_class, shape[-3], shape[-2], shape[-1]))

        for organ_index in range(1, num_class):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :, :] = temp_target
            # organ_target: (B, 14, 48, 128, 128)

        organ_target = organ_target.cuda()

        pt_1 = get_dice(pred, organ_target).mean()
        gamma = 0.75
        return torch.pow((2-pt_1), gamma)


