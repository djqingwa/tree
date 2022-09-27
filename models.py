from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":##是3合1 包括 卷积 batch norxxx + relu
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])#得到特征图的个数,第一个是32个后面就是32个通道,第二个是64的
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,#加了batch norxxx之后就不需要偏置向了
                ),
            )
            if bn:#batch normalization
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":#加一个relu激活函数
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":#V3中没有这个东西
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":#上采样,没有执行实际的操作,操作在后面
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route": # 输入1：26*26*256 输入2：26*26*128  输出：26*26*（256+128） #作一个拼接 把上采样13X13 变26X26 和原来的26X26拼接 
            #layers = -4#和向第数(自己也数一层)4层去做拼接操作
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":#残差连接,就是一个数值上的相加
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]#指定先验框的ID
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            #print(anchors)
            anchors = [anchors[i] for i in anchor_idxs]#根据ID取出先验框,如果是6,7,8取的是 (116, 90), (156, 198), (373, 326)]
            #print(anchors)

            num_classes = int(module_def["classes"])#输入的种类
            img_size = int(hyperparams["height"])#图像的大小
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)#构建yolo层
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)#最终输出的特征图个数加进来

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors#先验框的大小
        self.num_anchors = len(anchors)#先验框的个数
        self.num_classes = num_classes#要分类的个数
        self.ignore_thres = 0.5#阈值
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # Tensors for cuda support
        print (x.shape)#[4, 255, 13, 13]) 4:表示当前batch是等于4的,设多大跟显存相关的 255:我们现在得到的值有255个是通道数 当前特征图是13X13
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor#设置是GPU还是CPU跑,设置格式
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim#图片大小是随机的,最大500多,最小300多,只要这个值能被32整除就行
        num_samples = x.size(0)#4是batch表示我们一次训练4张图像 torch.Size([4, 255, 13, 13])
        grid_size = x.size(2)#grid_size是根据输入图像来的,输入图像多大,再除以一个32就完事了416/32=13

        prediction = (#终最要预测的结果,255是怎么来的 num_samples就是batch size self.num_anchors:候选框的个数
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)#把我的数据变换一个维度
            .permute(0, 1, 3, 4, 2)#调整顺利4 3 85 13 13 调整成 4 3 13 13 85
            .contiguous()#拷贝一份,x没有影响还是原来的数据
        )
       # torch.Size([4, 255, 13, 13])转成torch.Size([4, 3, 13, 13, 85]):3是三个候选框
        print (prediction.shape)
        # Get outputs
        #print(prediction[..., 0])  #85个 x y w h 致信度 80
        x = torch.sigmoid(prediction[..., 0])  # Center x#中心点的坐标  #怎么会那么多的数据????
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf 致信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.对于每一个类别我都需要知道属于当前类别的概率

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda) #相对位置得到对应的绝对位置比如之前的位置是0.5,0.5变为 11.5，11.5这样的

        # Add offset and scale with anchors #特征图中的实际位置
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w#torch.exp
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat( #还原成真实图
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride, #还原到原始图中
                pred_conf.view(num_samples, -1, 1),#致信度
                pred_cls.view(num_samples, -1, self.num_classes),#类别
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:#根据targets标签进行实际损失的计算
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(#标签的转换
                pred_boxes=pred_boxes,#x,y是相对于一个格子来说他的相对位子
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            # iou_scores：真实值与最匹配的anchor的IOU得分值 class_mask：分类正确的索引  obj_mask：目标框所在位置的最好anchor置为1 noobj_mask obj_mask那里置0，还有计算的iou大于阈值的也置0，其他都为1 tx, ty, tw, th, 对应的对于该大小的特征图的xywh目标值也就是我们需要拟合的值 tconf 目标置信度
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask]) # 只计算有目标的  #obj_mask 在做预测的时候对每一个位置都进行了预测,但绝大部分可能都是背景,背景没用,不用算损失 obj_mask中目标唯1的才计算损失
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])#和真实值算损失值
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask]) #致信度 一个前景的 bce_loss:因为只有0或1两种可能性
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])#一个背景的
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj #有物体越接近1越好 没物体的越接近0越好 #两个合在一起就是致信度损失,其中即包括了前景也包括了背景,然后乘上一个相应的权重参数
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask]) #分类损失
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls #总损失

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)#把配置文件读进来
        self.hyperparams, self.module_list = create_modules(self.module_defs)#按照配置文件写的顺序,逐层把结构定义好
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]#yolo层相当于我们得到最终的结果了,以及损失值的计算
        self.img_size = img_size#图片的大小
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):#配置文件和数据读了之后进入前向传播
        img_dim = x.shape[2]
        loss = 0#开始损失是0
        layer_outputs, yolo_outputs = [], []#layer_outputs 当前层输出结果, yolo_outputs 以及yolo输出结果,把结果不断往这里传
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:#如果是三个中的其中一个,直接执行因为是现成的API
                x = module(x)#经过一次卷积得到结果
            elif module_def["type"] == "route":#拼接
                # print(layer_outputs[-3].shape)
                # print(layer_outputs[-4].shape)
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)#这里有一个并并没有拼接
                # print(x.shape)
            elif module_def["type"] == "shortcut":#残差连接,就是一个数值上的相加
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]#-1是最后一层,+从后数前面第几层
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)#x是前一层的结果 img_dim输入图像的大小
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
