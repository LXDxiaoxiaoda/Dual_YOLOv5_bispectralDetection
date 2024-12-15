# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr, time_sync

from utils.plots import feature_visualization
from PIL import Image
from torchvision import transforms


try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer  ch: [256, 512, 1024]
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export    # self.training = self.training | self.export   self.training，self.export任一为真，self.training为真
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  ### inference 训练完成和验证时候进入
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))   ### 输出结果

        # 注意：训练的时候和导出的时候，return x；训练完成和验证时候，return (torch.cat(z, 1), x)。
        # 别忘了，训练完成会调用一次test()，所以这时候是return (torch.cat(z, 1), x)，所以pt模型输出是接到一起的
        # 需要验证onnx模型，要使得 return (torch.cat(z, 1), x) ，然后调用我哪个脚本试一下就好了
        return x if self.training else (torch.cat(z, 1), x)
    
    # # 转换模型的时候使用
    # def forward(self, x):
    #     z = []  # inference output    这个变量没用上
    #     for i in range(self.nl):
    #         x[i] = self.m[i](x[i])  # conv
    #         x[i] = F.sigmoid(x[i])  # 对每一个输出添加sigmoid
    #         # x[i] = F.relu(x[i])  # 对每一个输出添加relu

    #     return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        # 确保yaml文件是一个字典
        if isinstance(cfg, dict):   # 是字典
            self.yaml = cfg  # model dict

        else:  # is *.yaml          # 不是字典，则转换
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict
            # print("YAML")
            # print(self.yaml)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist    model的定义！！！
        # print(self.model)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # print(m)

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # print("1, ch, s, s", 1, ch, s, s)
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            # m.stride = torch.Tensor([8.0, 16.0, 32.0])    ### 三个检测头用这个，原版
            # # m.stride = torch.Tensor([8.0, 16.0])            ### 两个检测头用这个
            if len(self.yaml['anchors']) == 3:      # self.yaml 是配置文件， self.model 是已经序列化的模型了，用self.yaml 好做判断，是一个字典结构
                m.stride = torch.Tensor([8.0, 16.0, 32.0])    ### 三个检测头用这个，原版
            elif len(self.yaml['anchors']) == 2:
                m.stride = torch.Tensor([8.0, 16.0])            ### 两个检测头用这个
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2, augment=False, profile=False, visualize = False):
        if augment: 
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, x2, profile, visualize)  # single-scale inference, train


    def forward_once(self, x, x2, profile=False, visualize = False):
        """

        :param x:          RGB Inputs
        :param x2:         IR  Inputs
        :param profile:
        :return:
        """
        y, dt, FLOPs, params = [], [], [], []  # outputs, 模型传播一次时间, FLOPs, 参数量
        # rgb_input = deepcopy(x) # 保存一份原始的rgb输入
        # for i, m in enumerate(self.model):
        for m in self.model:
            if m.f != -1 and m.f != -4:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:     # 和verbose差不多的，打印出模型详细信息。这里使用thop.profile每层都打印
                ### by LXD
                c = m == self.model[-1]  # is final layer, copy input as inplace fix    是最后一层，复制输入因为原地操作
                # 因为最后一层会有原地操作x，所以先复制一份x的副本，不然运行不正确
                if m.f == -4:   # 不管是RGB的输入还是IR的输入都转成中间变量xx
                    xx = x2
                else:
                    xx = x
                o = thop.profile(m, inputs=(xx.copy() if c else xx, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs = Macs * 2
                t = time_synchronized()
                for _ in range(10):     # 跑十次，获得一个稳定的时间
                    m(xx.copy() if c else xx)
                dt.append((time_synchronized() - t) * 100)  # (time_synchronized() - t) / 10 * 1000
                FLOPs.append(o)
                params.append(m.np)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")  # >:右对齐 10:字符宽度 s:字符串
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}') 
                if c:
                    logger.info(f"{sum(dt):10.2f} {sum(FLOPs):10.2f} {sum(params):>10}  Total")

            if m.f == -4:   # 这就是为什么from -4的原因
                x = m(x2)   # 以x2开始传播，后面都是ir输入的了，因为x更新了
            # elif m.f == -2:
            #     x = m(rgb_input)    # 利用原始rgb信息
            # elif i >= 0 and i <= 2: # 这里只在共享的时候进入
            #     x = m(x)
            #     x2 = m(x2)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output    如果共享参数，这里怎么办？
            # print(len(y))

            # 添加可视化代码 by LXD
            if visualize:    
                feature_visualization(x, m.type, m.i)

        if profile:
            logger.info('%.1fms total' % sum(dt))

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # print("ch", ch)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        ### 实质上这里的操作是根据配置表还原args，args是传入每个模块真正的参数
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR, MobileNet_Block]:

            if m is Focus:          # 这里就是为什么第一个模块只能是Focus的原因
                c1, c2 = 3, args[0]     # args[0] 是指output_channels，因为这是从字典读的
                # print("focus c2", c2)
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8) # Returns x evenly divisible by divisor 返回c2 * gw可以被8整除的数，若c2 * gw=64，则不变；若c2 * gw=63，则为64
                args = [c1, c2, *args[1:]]  # 主要看这个变量传入什么组成模型，因为后面用nn.Sequential构建序列，就是用这个变量
                # print("args:", args)
            elif m is Conv and args[0] == 64: # 修正：使第一个模块使用Conv进入这里
                c1, c2 = 3, args[0]    
                # print("focus c2", c2)
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:                   # 原版：第一个模块用Conv会进入这里，然后c1不为3，是上一个channel长度，所以会报错
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                # print("args:", args)
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f]) 
        elif m in (Add, Add_weight, DMAF):
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            # print("Add2 arg", args[0])
            args = [c2, args[1]]
        elif m in [GPT]:
            c2 = ch[f[0]]
            args = [c2]
        elif m is NiNfusion:
            c1 = sum([ch[x] for x in f])
            c2 = c1 // 2
            args = [c1, c2, *args]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2

        # ICAFusion的模块
        elif m in [TransformerFusionBlock, TransformerFusionBlock2, GPT1]:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        elif m in [TransformerFusionBlock1]:
            c2 = ch[f[4]]
            args = [c2, *args[1:]]
        elif m is Multi_attention_fusion_block:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]

        elif m is Illumination:
            c1 = ch[f]
            c2 = c1*4
            args = [c1, c2]
        elif m is Illumination_v6:
            c1 = ch[f[0]]
            c2 = c1*4
            args = [c1, c2]
        elif m in [Illumination_weight, Illumination_fusion]:
            c1, c2 = ch[f[2]], args[0]  # f[2]代表光照层，来自光照层的输出
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8) 

            args = [c1, c2, *args[1:]]
        # elif m is IAN:
        #     c1 = 3
        #     c2 = args[0]
        #     args = [c1, c2]
        elif m is random_weight:
            c1 = 256
            c2 = args[0]
            args = [c1, c2]

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module  通过args构建模型，m代表model
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        # if i == 4:
        #     ch = []
        ch.append(c2)
    # print(layers)
    return nn.Sequential(*layers), sorted(save) # 返回一个序贯模型，用于简单构建网络。


if __name__ == '__main__':
    flag = int(input("请输入0或1，0代表验证网络结构，1代表特征可视化："))

    parser = argparse.ArgumentParser()
    if flag == 0:
        parser.add_argument('--cfg', type=str, default='/project/multispectral-object-detection/models/my_test/yolov5l_illumination_FLIR_v7.yaml', help='model.yaml')   # 验证网络结构是否能用
    else:
        parser.add_argument('--weights', nargs='+', type=str, default='/project/multispectral-object-detection/runs/train/151/exp2/weights/best.pt', help='model.pt path(s)')  # 验证训练的权重文件
    parser.add_argument('--device', default='7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    set_logging()

    device = select_device(opt.device)
    print(device)


    if flag == 0:
        model = Model(opt.cfg).to(device)   # 使用网络结构实例化
        input_rgb = torch.Tensor(1, 3, 640, 640).to(device)     ### 8, 3, 640, 640 >> 1, 3, 640, 640
        input_ir = torch.Tensor(1, 3, 640, 640).to(device)
        # 1、服务器上不怕超出内存
        # 2、仅验证的不需要这么batch=1也OK
        # 3、输出GFLOPs时候是正确的，就是一张图片传输的计算量

        output = model(input_rgb, input_ir, profile = True)

        print("YOLO")
        print(output[0].shape)
        print(output[1].shape)
        print(output[2].shape)

    else:
        model = attempt_load(opt.weights, map_location=device)  # 使用权重文件实例化

        # 1、读取图片
        img_rgb = cv2.imread("/project/datasets/FLIR_aligned/visible/test/FLIR_08935.jpeg")
        img_ir = cv2.imread("/project/datasets/FLIR_aligned/infrared/test/FLIR_08935.jpeg")

        # 2、预处理
        img_rgb = letterbox(img_rgb, 640, stride=32, auto=False)[0]
        img_rgb = img_rgb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_rgb = np.ascontiguousarray(img_rgb)
        img_rgb = torch.from_numpy(img_rgb).to(device)  # numpy转tensor
        img_rgb = img_rgb.float()/255.0  # 0 - 255 to 0.0 - 1.0  归一化
        img_rgb = img_rgb.unsqueeze(0)  # 扩展一维

        img_ir = letterbox(img_ir, 640, stride=32, auto=False)[0]
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_ir = np.ascontiguousarray(img_ir)
        img_ir = torch.from_numpy(img_ir).to(device)
        img_ir = img_ir.float()/255.0  # 0 - 255 to 0.0 - 1.0
        img_ir = img_ir.unsqueeze(0)

        print(img_rgb.shape)    # torch.Size([1, 3, 640, 640])

        # 3、可视化特征
        output = model(img_rgb, img_ir, visualize = True)   
        print("Done")


    # output = model(input_rgb, input_ir) # 以函数的方式调用实例，__call__方法在基类nn.Module中定义，这里__call__指定调用forward方法
    # output = model(input_rgb, input_ir, profile = True) # 有两个参数，augment 增强，profile 打印模型详情

    
