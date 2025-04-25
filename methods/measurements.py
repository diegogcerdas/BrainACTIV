import numpy as np
import torch
import visualpriors
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from methods.xtc_network import UNet


class DepthEstimator:

    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        self.zoe = self.zoe.to(device).eval()
        self.device = device

    def compute(self, img):
        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)).unsqueeze(0).float().to(self.device) / 255
        with torch.no_grad():
            depth = self.zoe.infer(img_tensor).squeeze().detach().cpu().numpy()
            depth = depth[None,:,:]
        return depth
    
class SurfaceNormalEstimator:

    def __init__(self, model_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.normal_model = UNet() 
        self.normal_model.load_state_dict(torch.load(model_path, map_location=device))
        self.normal_model = self.normal_model.to(device).eval()
        self.device = device

    def compute(self, img):
        img_tensor = torch.tensor(np.array(img.resize((256, 256))).transpose(2, 0, 1)).unsqueeze(0).float().to(self.device) / 255
        with torch.no_grad():
            normal = self.normal_model(img_tensor)
            normal = torch.nn.functional.interpolate(
                normal,
                size=img.size,
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).clamp(min=0, max=1)
            normal = normal.squeeze().permute(1,2,0).detach().cpu().numpy()
            normal = np.moveaxis(normal, -1, 0)
        return normal
    
class CurvatureEstimator:

    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device

    def compute(self, img):
        img_tensor = torch.tensor(np.array(img.resize((256, 256))).transpose(2, 0, 1)).unsqueeze(0).float().to(self.device) / 255
        principal_curvature = (visualpriors.feature_readout(img_tensor * 2 - 1, 'curvature', device=self.device) / 2. + 0.5)[:,:2]
        principal_curvature = torch.nn.functional.interpolate(
            principal_curvature,
            size=img.size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).clamp(min=0, max=1)
        principal_curvature = principal_curvature.squeeze().permute(1,2,0).detach().cpu().numpy()
        gaussian_curvature = np.prod(principal_curvature, -1)[None,:,:]
        return gaussian_curvature

def compute_warmth(img):
    hue = np.array(img.convert('HSV'))[:,:,[0]]
    saturation = np.array(img.convert('HSV'))[:,:,[1]]
    value = np.array(img.convert('HSV'))[:,:,[2]]
    measure = np.cos(hue/255*np.pi*2) * (saturation / 255) * (value / 255)
    measure = ((measure + 1) / 2)
    measure = np.moveaxis(measure, -1, 0)
    return measure

def compute_saturation(img):
    measure = np.array(img.convert('HSV'))[:,:,[1]]
    measure = np.moveaxis(measure, -1, 0) / 255
    return measure

def compute_brightness(img):
    measure = np.array(img.convert('HSV'))[:,:,[2]]
    measure = np.moveaxis(measure, -1, 0) / 255
    return measure

def compute_entropy(img):
    image = img_as_ubyte(np.array(img.convert('L')))
    measure = entropy(image, disk(5)) / np.log2(256)
    measure = measure[None,:,:]
    return measure