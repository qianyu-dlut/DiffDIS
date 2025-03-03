import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import random
from .image_util import resize_max_res_tensor

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def check_mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def replace_unet_conv_in(unet):
        _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        _weight *= 0.5
        _n_convin_out_channel = unet.conv_in.out_channels
        _new_conv_in = nn.Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        unet.conv_in = _new_conv_in
        print("Unet conv_in layer is replaced")
        unet.config["in_channels"] = 8
        print("Unet config is updated")
        return unet


def update_att_weights(unet):
    for key, param in unet.named_parameters():
        if 'attn3' in key:
            new_key = key.replace('attn3', 'attn1')
            if new_key in unet.state_dict():
                param.data.copy_(unet.state_dict()[new_key].data)
        elif 'norm2_' in key:
            new_key = key.replace('norm2_', 'norm1')
            param.data.copy_(unet.state_dict()[new_key].data)
    return unet


def pyramid_noise_like(x, discount):
    b, c, w, h = x.shape
    u = nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random()*2+2 
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
        if w==1 or h==1: break 
    return noise/noise.std() 



def generate_multi_scale_latents(rgb_mix, scale_factor, vae, weight_dtype, opt):
    def get_latents(resolution):
        moments = vae.quant_conv(vae.encoder(
            resize_max_res_tensor(rgb_mix, is_disp=False, recom_resolution=resolution).cuda().to(weight_dtype)
        ))
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean * scale_factor

    latents_2 = get_latents(opt.trainsize // 2).repeat(2, 1, 1, 1)
    latents_4 = get_latents(opt.trainsize // 4).repeat(2, 1, 1, 1)
    latents_8 = get_latents(opt.trainsize // 8).repeat(2, 1, 1, 1)

    return latents_2, latents_4, latents_8


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]
        #print(c)
        #d = torch.mean(torch.stack(c))
        #print(d)
        return torch.mean(torch.stack(c))



def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)



