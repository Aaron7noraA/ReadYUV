import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from math import log10
import torch.nn as nn
import csv
import argparse
from Setting import DATASET
import os

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader



################## Matrix for MS-SSIM ##################
@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    """create 1D gauss kernel  
    Args:
        window_size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
        channel (int): input channel
    """
    half_window = window_size // 2
    coords = torch.arange(-half_window, half_window+1).float()

    g = (-(coords ** 2) / (2 * sigma ** 2)).exp_()
    g.div_(g.sum())

    return g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)

@torch.jit.script
def gaussian_blur(x, window, use_padding: bool):
    """Blur input with 1-D gauss kernel  
    Args:
        x (tensor): batch of tensors to be blured
        window (tensor): 1-D gauss kernel
        use_padding (bool): padding image before conv
    """
    C = x.size(1)
    padding = 0 if not use_padding else window.size(3) // 2
    out = F.conv2d(x, window, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window.transpose(2, 3),
                   stride=1, padding=(padding, 0), groups=C)
    return out

@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    """Calculate ssim for X and Y  
    Args:
        X (tensor):Y (tensor): a batch of images, (N, C, H, W)
        window (tensor): 1-D gauss kernel
        data_range (float): value range of input images. (usually 1.0 or 255)
        use_padding (bool, optional): padding image before conv. Defaults to False.
    """
    K1, K2 = 0.01, 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = gaussian_blur(X, window, use_padding)
    mu2 = gaussian_blur(Y, window, use_padding)
    sigma1_sq = gaussian_blur(X * X, window, use_padding)
    sigma2_sq = gaussian_blur(Y * Y, window, use_padding)
    sigma12 = gaussian_blur(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim, cs

@torch.jit.script
def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False):
    """Calculate ms_ssim for X and Y  
    Args:
        X (tensor):Y (tensor): a batch of images, (N, C, H, W)
        window (tensor): 1-D gauss kernel
        data_range (float): value range of input images. (usually 1.0 or 255)
        weights (tensor): weights for different levels
        use_padding (bool, optional): padding image before conv. Defaults to False.
    """
    css, ssims = [], []
    for _ in range(weights.size(0)):
        ssim, cs = ssim(X, Y, window, data_range, use_padding)
        css.append(cs)
        ssims.append(ssim)
        padding = (X.size(-2) % 2, X.size(-1) % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ms_css = torch.stack(css[:-1], dim=0) ** weights[:-1].unsqueeze(1)
    ms_ssim = torch.prod(ms_css * (ssims[-1] ** weights[-1]), dim=0)
    return ms_ssim

class SSIM(torch.jit.ScriptModule):
    """Structural Similarity index  
    Args:
        window_size (int, optional): the size of gauss kernel. Defaults to 11.
        window_sigma (float, optional): sigma of normal distribution. Defaults to 1.5.
        data_range (float, optional): value range of input images. (usually 1.0 or 255). Defaults to 255..
        channel (int, optional): input channels. Defaults to 3.
        use_padding (bool, optional): padding image before conv. Defaults to False.
        reduction (str, optional): reduction mode. Defaults to "none".
    """
    __constants__ = ['data_range', 'use_padding', 'reduction']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False, reduction="none"):
        super().__init__()
        self.data_range = data_range
        self.use_padding = use_padding
        self.reduction = reduction

        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)

    @torch.jit.script_method
    def forward(self, input, target):
        ret = ssim(input, target, self.window,
                   self.data_range, self.use_padding)[0]
        if self.reduction != 'none':
            ret = ret.mean() if self.reduction == 'mean' else ret.sum()
        return ret

class MS_SSIM(SSIM):
    """Multi-Scale Structural Similarity index  
    Args:
        window_size (int, optional): the size of gauss kernel. Defaults to 11.
        window_sigma (float, optional): sigma of normal distribution. Defaults to 1.5.
        data_range (float, optional): value range of input images. (usually 1.0 or 255). Defaults to 255..
        channel (int, optional): input channels. Defaults to 3.
        use_padding (bool, optional): padding image before conv. Defaults to False.
        reduction (str, optional): reduction mode. Defaults to "none".
        weights (list of float, optional): weights for different levels. Default to [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        levels (int, optional): number of downsampling
    """
    __constants__ = ['data_range', 'use_padding', 'reduction']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False, reduction="none",
                 weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], levels=None):
        super().__init__(window_size, window_sigma,
                         data_range, channel, use_padding, reduction)

        weights = torch.FloatTensor(weights)
        if levels is not None:
            weights = weights[:levels]
            weights /= weights.sum()
        self.register_buffer('weights', weights)

    @torch.jit.script_method
    def forward(self, input, target):
        ret = ms_ssim(input, target, self.window, self.data_range,
                      self.weights, self.use_padding)
        if self.reduction != 'none':
            ret = ret.mean() if self.reduction == 'mean' else ret.sum()
        return ret



################## Color Transformation ##################

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722),
    "ITU-R_BT.601": (0.299, 0.587, 0.114)
}

def ycbcr444_to_rgb_Tensor(y, uv):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    # uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    y = y.unsqueeze(0)
    uv = uv.unsqueeze(0)
    
    y = y - (16/256)
    uv = uv - (128/256)
    yuv = torch.cat([y, uv], dim=1)
    
    T = torch.FloatTensor([[ 0.257,  0.504,   0.098],
                        [-0.148, -0.291,   0.439],
                        [ 0.439, -0.368,  -0.071]]).to(y.device)
    T = torch.linalg.inv(T)
    rgb = T.expand(yuv.size(0), -1, -1).bmm(yuv.flatten(2)).view_as(yuv)
    return rgb.clamp(min=0, max=1)[0]

def ycbcr444_to_rgb_BT709(y, uv):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    y  = y.numpy()
    uv = uv.numpy()
    # uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    cb = uv[0:1, :, :]
    cr = uv[1:2, :, :]
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return torch.from_numpy(rgb)

def rgb_to_ycbcr_bt709(rgb):
    """
    rgb: 3 x H x W tensor in [0,1] (full-range)
    returns: 3 x H x W tensor [Y, Cb, Cr] in [0,1]
    """
    Kr, Kb = 0.2126, 0.0722
    Kg = 1 - Kr - Kb

    r, g, b = rgb[0], rgb[1], rgb[2]

    y  = Kr * r + Kg * g + Kb * b
    cb = (b - y) / (2 * (1 - Kb)) + 0.5
    cr = (r - y) / (2 * (1 - Kr)) + 0.5

    ycbcr = torch.stack([y, cb, cr], dim=0)
    return ycbcr.clamp(0,1)


def rgb_to_yuv444_bt601(rgb):
    # rgb: 3xhxw float tensor in [0,1]
    T = torch.tensor([
        [ 0.257,  0.504,  0.098],
        [-0.148, -0.291,  0.439],
        [ 0.439, -0.368, -0.071]
    ], dtype=rgb.dtype, device=rgb.device)

    offset = torch.tensor([16/255, 128/255, 128/255],
                          dtype=rgb.dtype, device=rgb.device).view(3,1,1)

    yuv = T @ rgb.flatten(1)   # shape (3, H*W)
    yuv = yuv.view(3, *rgb.shape[1:]) + offset
    return yuv.clamp(0,1)

################## Color Transformation ##################


class Read_YUV_Video():
    def __init__(self, file_path, in_format, out_format, H_W, frame_num, interpolation='bilinear', bitdepth=8):
        """
        Args:
            file_path (str): video path
            in_format (str): yuv sub sampling format for input video, 444, 420
            out_format (str): yuv sub sampling format for out video, 444, 420
            H_W (tuple):  (H, W)
            frame_num (int):  frame number to be coded
        """
        assert bitdepth in [8, 10], "There is no such bitdepth configuration !!!"
        
        self.files = open(file_path, "rb")
        self.in_format = in_format
        self.out_format = out_format
        self.H, self.W = H_W
        # self.H = (self.H // 64) * 64
        # self.W = (self.W // 64) * 64
        self.frame_num = frame_num
        self.interpolation = interpolation
        self.iter_cnt = 1
        
        
        # Y components
        self.y_size = self.H * self.W
        
        # UV components
        self.scale = 1 if in_format == '444' else 2
        self.uv_size = self.H * self.W * 2 if in_format == '444' else self.H * self.W // self.scale
        
        self.bitdepth = bitdepth
        self.dtype = np.uint8
        self.max_val = (1 << self.bitdepth) - 1
        if bitdepth == 10:
            self.y_size *= 2
            self.uv_size *= 2
            self.dtype = np.uint16
        
        # check video frame number
        self.check_video_sanity(file_path)
        
        
    def read_one_frame(self):
        if self.iter_cnt > self.frame_num:
            raise "Access frame out of range. Accessing {self.iter_cnt}th frame, but frame range set to {self.frame_num} !!!"
        
        self.iter_cnt += 1
        
        Y = self.files.read(self.y_size)
        UV = self.files.read(self.uv_size)
        Y = np.frombuffer(Y, dtype=self.dtype).reshape(1, self.H, self.W)
        UV = np.frombuffer(UV, dtype=self.dtype).reshape(2, self.H // self.scale, self.W // self.scale)
        Y = Y.astype(np.float32) / self.max_val
        UV = UV.astype(np.float32) / self.max_val

        Y = torch.from_numpy(Y).type(torch.FloatTensor)    
        UV = torch.from_numpy(UV).type(torch.FloatTensor)  
        
        if self.in_format == self.out_format:
            return Y, UV
        elif self.in_format == '420' and self.out_format == '444':
            UV = F.interpolate(UV.unsqueeze(0), scale_factor=2, mode=self.interpolation)[0]
            return Y, UV
        elif self.in_format == '444' and self.out_format == '420':
            UV = F.interpolate(UV.unsqueeze(0), scale_factor=1/2, mode=self.interpolation)[0]
            return Y, UV
        else:
            raise NotImplementedError
        
    def check_video_sanity(self, file_path):
        total_size = Path(file_path).stat().st_size
        if total_size < (self.y_size + self.uv_size) * self.frame_num:
            raise "Video Frame Out of Range !!!"
        
class MatrixTYPE():
    def __init__(self, TYPE='rgb'):
        self.criterion = MS_SSIM(data_range=1.) if TYPE == 'ssim' else nn.MSELoss(reduction='none') 

    def mse2psnr(self, mse, data_range=1.):
        """PSNR for numpy mse"""
        return 20 * log10(data_range) - 10 * log10(mse)
    
    def compute_mse_rgb(self, rec, raw):
        return self.criterion(rec, raw).mean()
    
    def compute_PSNR_rgb(self, rec, raw):
        mse = self.criterion(rec, raw).mean()
        return self.mse2psnr(mse)
    
    def compute_MSSSIM_rgb(self, rec, raw):
        return self.criterion(rec, raw).mean().item()
    
    def compute_PSNR_YUV(self, rec_y, rec_uv, raw_y, raw_uv):
        
        yuv_mse, y_mse, u_mse, v_mse = self.compute_mse(rec_y, rec_uv, raw_y, raw_uv)
        y_psnr = self.mse2psnr(y_mse)
        u_psnr = self.mse2psnr(u_mse)
        v_psnr = self.mse2psnr(v_mse)
        return (y_psnr * 6 + u_psnr + v_psnr) / 8, y_psnr, u_psnr, v_psnr

    def compute_mse(self, rec_y, rec_uv, raw_y, raw_uv):

        rec_u = rec_uv[0]
        rec_v = rec_uv[1]
        
        raw_u = raw_uv[0]
        raw_v = raw_uv[1]
        
        y_mse = self.criterion(rec_y, raw_y).mean()
        u_mse = self.criterion(rec_u, raw_u).mean()
        v_mse = self.criterion(rec_v, raw_v).mean()
        
        return (y_mse * 6 + u_mse + v_mse) / 8, y_mse, u_mse, v_mse 


def Compute_RD_PerSequence(rec_video_path, rec_cfg_path, gt_video_path, report_path, H_W=(1080, 1920), frame_num=96, TYPE='rgb', format='601', in_form='444', out_form='444', crop=0, bitdepth=8):
    if crop:
        H_W = ((H_W[0]//crop)*crop, (H_W[1]//crop)*crop)

    if H_W[0] > H_W[1]: # the order should be (H, W)
        H_W = (H_W[1], H_W[0])
        
    def getBits(rec_cfg_path, length=96):
        bits = [0] * length
        with open(rec_cfg_path, 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                if "POC" in line:
                    idx = int(line.split("TId:")[0].split()[1].strip())
                    if idx < length:
                        bits[idx] = int(line.split(")")[-1].split("bits")[0].strip())
        return bits
    

    rec_video = Read_YUV_Video(file_path=rec_video_path, in_format=in_form, out_format=out_form, H_W=H_W, frame_num=frame_num, bitdepth=bitdepth) 
    raw_video = Read_YUV_Video(file_path=gt_video_path, in_format=in_form, out_format=out_form, H_W=H_W, frame_num=frame_num, bitdepth=bitdepth)  
    Matrix = MatrixTYPE(TYPE=TYPE)
    
    bits_profile = [ i / (H_W[0] * H_W[1]) for i in getBits(rec_cfg_path)]
    
    
    transformer = transforms.Compose([
                transforms.ToTensor()
            ])

    if TYPE == 'rgb':
        columns = ['frame', 'PSNR', 'Rate']
    elif TYPE == 'yuv':
        columns = ['frame', 'YUV-PSNR', 'Rate', 'Y-PSNR', 'U-PSNR', 'V-PSNR']
    elif TYPE == 'ssim':
        columns = ['frame', 'MS-SSIM', 'Rate']
    else:
        raise NotImplementedError
    
    with open(report_path, 'w', newline='') as report:
        writer = csv.writer(report, delimiter=',')
        writer.writerow(columns)
            
        for idx in range(frame_num):
            rec_y, rec_uv = rec_video.read_one_frame()
            raw_y, raw_uv = raw_video.read_one_frame()
            
            if TYPE != 'yuv':
                if format == '601':
                    raw_frame = ycbcr444_to_rgb_Tensor(raw_y, raw_uv)
                    rec_frame = ycbcr444_to_rgb_Tensor(rec_y, rec_uv)
                elif format == '709':
                    raw_frame = ycbcr444_to_rgb_BT709(raw_y, raw_uv)
                    rec_frame = ycbcr444_to_rgb_BT709(rec_y, rec_uv)
                else:
                    raise NotImplementedError
            
            # raw_frame = transformer(imgloader('/work/DATASET/TestVideo/raw_video_1080/HEVC-B/BasketballDrive/frame_1.png'))
            # print(raw_frame.shape)
            # print(frame.shape)
            # print((raw_frame - frame).abs)
            # save_image(frame, 't.png')
            # save_image(raw_frame, 't2.png')
            # exit()
            
            if TYPE == 'rgb':
                PSNR = Matrix.compute_PSNR_rgb(rec_frame, raw_frame)
                writer.writerow([f'frame_{idx + 1}', PSNR, bits_profile[idx]])
            elif TYPE == 'yuv':
                YUV_PSNR, Y_PSNR, U_SNR, V_PSNR = Matrix.compute_PSNR_YUV(rec_y, rec_uv, raw_y, raw_uv)
                writer.writerow([f'frame_{idx + 1}', YUV_PSNR, bits_profile[idx], Y_PSNR, U_SNR, V_PSNR])
            elif TYPE == 'ssim':
                SSIM = Matrix.compute_MSSSIM_rgb(rec_frame.unsqueeze(0), raw_frame.unsqueeze(0))
                writer.writerow([f'frame_{idx + 1}', SSIM, bits_profile[idx]])
            else: 
                raise NotImplementedError
        
    
        
            
        
        


if __name__ == '__main__':
    
    print('Remember to check Dataset Profile is as expected !!!')
    datasets = DATASET
    parser = argparse.ArgumentParser(description="Select which dataset to test on VTM17.0 codec")
    parser.add_argument('--datasetRoot', type=str, required=True)
    parser.add_argument('--recRoot', type=str, default="./LDP")
    parser.add_argument('--savePath', type=str, required=True)
    parser.add_argument('--format', type=str, default="601")
    parser.add_argument('--in_form', type=str, default="444")
    parser.add_argument('--out_form', type=str, default="444")
    parser.add_argument('--matrix', type=str, default="rgb")
    parser.add_argument('--QP_LIST',  type=str, nargs='+', default=[])
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--bitdepth', type=int, default=8)
    args = parser.parse_args()
    
    
    save_root_base = args.savePath
    os.makedirs(save_root_base, exist_ok=True)
    
    for dataset in DATASET.keys():
        for QP in args.QP_LIST:
            save_root = os.path.join(save_root_base, f"qp={QP}", 'report')
            os.makedirs(save_root, exist_ok=True)
            for seq in DATASET[dataset].keys():
                Compute_RD_PerSequence(
                    rec_video_path= os.path.join(args.recRoot, dataset, f"qp={QP}", seq, f"{DATASET[dataset][seq]['vi_name']}.yuv"),
                    rec_cfg_path=  os.path.join(args.recRoot, dataset, f"qp={QP}", seq, f"{DATASET[dataset][seq]['vi_name']}.out"),
                    gt_video_path= os.path.join(args.datasetRoot, dataset, seq, f"{DATASET[dataset][seq]['vi_name']}.yuv"),
                    report_path= os.path.join(save_root, f"{seq}.csv"),
                    H_W=DATASET[dataset][seq]['frameWH'],
                    frame_num= 96,
                    TYPE=args.matrix,
                    format=args.format,
                    in_form=args.in_form,
                    out_form=args.out_form,
                    crop=args.crop,
                    bitdepth=args.bitdepth
                )