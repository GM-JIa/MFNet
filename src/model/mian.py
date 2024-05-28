import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from pyclbr import Function
import pywt




class CTU(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, dim=512):
        super(CTU, self).__init__()
        mid_channels = 1
        self.branch1 = nn.Sequential(
            BasicConv(in_channels, mid_channels, kernel_size=3, padding=1),
            TransAttn(img_size, mid_channels, patch_size, dim),
            BasicConv(mid_channels, mid_channels, kernel_size=3, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, mid_channels, kernel_size=5, padding=2),
            ResConv(mid_channels, mid_channels),
            BasicConv(mid_channels, mid_channels, kernel_size=5, padding=2)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_channels, mid_channels, kernel_size=7, padding=3),
            TransAttn(img_size, mid_channels, patch_size, dim),
            BasicConv(mid_channels, mid_channels, kernel_size=3, padding=1),
        )
        self.final_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = x1 + x2 + x3
        out = self.final_conv(out)
        return out


class GCM(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, dim=512):
        super(GCM, self).__init__()
        mid_channels = 4
        self.CTU1 = CTU(in_channels, mid_channels, img_size, patch_size, dim)
        self.CTU2 = CTU(mid_channels, mid_channels, img_size, patch_size, dim)
        self.CTU3 = CTU(mid_channels, out_channels, img_size, patch_size, dim)
    def forward(self, x):
        x = self.CTU1(x)
        x = self.CTU2(x)
        x = self.CTU3(x)
        return x


class SpatialFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialFusion, self).__init__()

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(out_channels)
        )

        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        avg_out = torch.mean(y2, dim=1, keepdim=True)
        max_out, _ = torch.max(y2, dim=1, keepdim=True)
        y2 = torch.cat([avg_out, max_out], dim=1)
        y2 = self.conv3(y2)
        y3 = y1 * y2
        out = self.conv4(y3)
        return out


class ChannelFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelFusion, self).__init__()

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(out_channels)
        )
    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        avg_out = self.conv2(self.avg_pool(x2))
        max_out = self.conv2(self.max_pool(x2))
        y2 = self.sigmoid(avg_out + max_out)
        y3 = y1 * y2
        out = self.conv3(y3)
        return out

class WaveletTransformFunction(Function):
    @staticmethod
    def forward(ctx, input, wavelet='db1'):
        ctx.wavelet = wavelet
        coeffs = pywt.wavedec2(input.cpu().numpy(), wavelet=wavelet, level=1)
        cA, (cH, cV, cD) = coeffs
        cA, cH, cV, cD = torch.from_numpy(cA).cuda(), torch.from_numpy(cH).cuda(), torch.from_numpy(cV).cuda(), torch.from_numpy(cD).cuda()
        ctx.save_for_backward(cA, cH, cV, cD)
        return cA, cH, cV, cD
    @staticmethod
    def backward(ctx, grad_cA, grad_cH, grad_cV, grad_cD):
        cA, cH, cV, cD = ctx.saved_tensors
        grad_cA, grad_cH, grad_cV, grad_cD = grad_cA.cpu().numpy(), grad_cH.cpu().numpy(), grad_cV.cpu().numpy(), grad_cD.cpu().numpy()
        grad_input = pywt.waverec2((grad_cA, (grad_cH, grad_cV, grad_cD)), wavelet=ctx.wavelet)
        grad_input = torch.from_numpy(grad_input).cuda()
        return grad_input, None

class WaveletBlock(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletBlock, self).__init__()
        self.wavelet = wavelet
    def forward(self, input):
        return WaveletTransformFunction.apply(input, self.wavelet)

class WaveletNet(nn.Module):
    def __init__(self, in_channels, out_channels, wavelet='db1'):
        super(WaveletNet, self).__init__()
        self.wavelet_transform = WaveletBlock(wavelet)
        self.hight_conv = nn.Conv2d(3*in_channels, out_channels, kernel_size=3, padding=1)
        self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, input):
        cA, cH, cV, cD = self.wavelet_transform(input)
        hight_input = torch.cat((cH, cV, cD), dim=1)
        low_input = cA
        hight_output = self.hight_conv(hight_input)
        low_output = self.low_conv(low_input)
        return hight_output, low_output


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class TransAttn(nn.Module):
    def __init__(self, img_size, mid_channels, patch_size=8, dim=1024, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        h = int(img_size / patch_size)
        self.img_to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * mid_channels, dim)
        )
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.patch_to_img = nn.Sequential(
            nn.Linear(dim, patch_size * patch_size * mid_channels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
        )

    def forward(self, x):
        x = self.img_to_patch(x)
        x = self.attn(x)
        x = self.patch_to_img(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResConv, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
    def forward(self, x):
        out = self.left(x)
        out += x
        out = F.relu(out)
        return out



