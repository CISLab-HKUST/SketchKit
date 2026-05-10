import os,cv2,time
import random,datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torch.hub import load_state_dict_from_url

DEFAULT_WEIGHTS_DIR = os.path.expanduser("~/.sketchkit/weights/LineDrawer")
WEIGHTS_URL_BASE = "https://cislab.hkust-gz.edu.cn/projects/linedrawer/weights/"


def _resolve_weight_path(model_name: str, suffix: str = "") -> str:
    model_name = os.fspath(model_name)
    base_name = os.path.basename(model_name)
    filename = base_name if base_name.endswith(".pth") else f"{base_name}{suffix}.pth"
    if os.path.isdir(model_name):
        return os.path.join(model_name, filename)
    if os.path.isabs(model_name) or os.path.sep in model_name or os.path.exists(model_name):
        return model_name if model_name.endswith(".pth") else f"{model_name}{suffix}.pth"
    return os.path.join(DEFAULT_WEIGHTS_DIR, filename)


def _resolve_weight_url(model_name: str, suffix: str = "") -> str:
    base_name = os.path.basename(os.fspath(model_name))
    filename = base_name if base_name.endswith(".pth") else f"{base_name}{suffix}.pth"
    return f"{WEIGHTS_URL_BASE}{filename}"


def _load_weight_state_dict(model_name: str, suffix: str, device):
    model_path = _resolve_weight_path(model_name, suffix)
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        return torch.load(model_path, map_location=device)

    model_url = _resolve_weight_url(model_name, suffix)
    print(f"Loading weights from {model_url}")
    return load_state_dict_from_url(model_url, map_location=device)


def rect_boudary(t_big_img ,rect):#paste_back
    """Clips a rectangle to the boundaries of a given image tensor.

    Args:
        t_big_img (torch.Tensor): The reference image tensor, used to determine
            boundaries. Shape is expected to be (B, C, H, W).
        rect (tuple): A tuple (x, y, w, h) representing the rectangle.

    Returns:
        tuple: A tuple (nx, ny, nw, nh) representing the clipped rectangle,
            ensuring it does not go out of the image bounds.
    """
    x,y,w,h=rect
    ih,iw=t_big_img.shape[-2:]
    #rect_boudary 1024 1024 torch.Size([1, 1, 1024, 1024]) print('rect_boudary',ih,iw,t_big_img.shape)
    nx=max(x,0)
    ny=max(y,0)
    nw= min(w,w+x,iw-x)#[w+x if nx<0 else min(w,iw-x)][0]#min(w,w+x,iw-x)
    nh= min(h,h+y,ih-y)#[h+y if nx<0 else min(h,ih-y)][0]#min(h,h+y,ih-y)

    return nx,ny,nw,nh

def pad_back(t_big_img,t_mask,rect):
    """Pastes a small mask tensor back into a full-sized tensor at a given rectangle.

    The mask is placed according to the `rect` coordinates. If the `rect`
    is partially outside the `t_big_img` boundaries, it's clipped.
    The final mask is also filtered so that it's zero wherever `t_big_img` is zero.

    Args:
        t_big_img (torch.Tensor): The original full-sized image tensor.
        t_mask (torch.Tensor): The small mask tensor to paste.
        rect (tuple): The (x, y, w, h) coordinates where the `t_mask`
            was originally extracted from.

    Returns:
        torch.Tensor: A full-sized mask tensor with `t_mask` pasted into it.
    """
    x,y,w,h=rect
    big_mask=torch.zeros(t_big_img.shape)
    nx,ny,nw,nh=rect_boudary(t_big_img,rect)

    big_mask[:,:,ny:ny+nh,nx:nx+nw]=t_mask[:,ny-y:ny-y+nh,nx-x:nx-x+nw]
    big_mask[t_big_img==0]=0
    return big_mask

def pad_back_lst(t_big_img,t_masks,rect_lst):
    """Pads back a batch of masks to their original full-size locations.

    Applies the `pad_back` function to each mask and rectangle in the provided
    lists and concatenates the results into a single batch tensor.

    Args:
        t_big_img (torch.Tensor): The original full-sized image tensor.
        t_masks (list[torch.Tensor]): A list of small mask tensors.
        rect_lst (list[tuple]): A list of (x, y, w, h) rectangles corresponding
            to each mask.

    Returns:
        torch.Tensor: A batch of full-sized masks (concatenated on dim=0).
    """
    lst=[]
    for i in range(len(rect_lst)):
        lst.append(pad_back(t_big_img.detach(),t_masks[i],rect_lst[i]))
    return torch.cat(lst,dim=0)    

class stNet(nn.Module):
    """A single-stage convolutional neural network.

    This network appears to be an encoder-decoder architecture, using
    strided convolutions for downsampling and PixelShuffle for upsampling.

    Attributes:
        DEVICE (str): The device to run the model on ('cpu' or 'cuda').
        preprocess (transforms.Compose): Image preprocessing transforms.
        conv (nn.Sequential): The main convolutional network structure.
    """
    def __init__(self,DEVICE='cpu'):
        """Initializes the stNet model.

        Args:
            DEVICE (str, optional): The device to run the model on ('cpu' or 'cuda').
                Defaults to 'cpu'.
        """
        super(stNet, self).__init__()
        N_in_channel,N_out_channel=2,1
        self.DEVICE=DEVICE
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.conv = nn.Sequential(
            nn.Conv2d(N_in_channel, 128, kernel_size=9, stride=2, padding=4, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #conv x4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),        
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            #conv x3
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),  
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),  
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(), 

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(), 
            #conv x2
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(), 
            #conv x6
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            #20% spatial dropout
            nn.Dropout(p=0.2),   

            nn.PixelShuffle(upscale_factor=2),   
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            nn.PixelShuffle(upscale_factor=2),   
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16), 
            nn.PixelShuffle(upscale_factor=2), 
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid() 
            )
        
            
    @torch.no_grad()   
    def center_crop(self,input_c2,rect):
        """Crops a tensor to a specified rectangle.

        Wrapper for `torchvision.transforms.functional.crop`.

        Args:
            input_c2 (torch.Tensor): The input tensor to crop.
            rect (tuple): A tuple (i, j, w, h) defining the crop box.
                Note: `i` is x (width-wise) and `j` is y (height-wise).

        Returns:
            torch.Tensor: The cropped tensor.
        """
        i, j, w, h=rect
        if i<0 or j<0:
            pass
            #print(i, j, h, w)
        input_c2 = TF.crop(input_c2, j, i, h, w)
        #print(center_crop,rect,input_c2.shape)
        return input_c2
    
    @torch.no_grad()
    def forward(self, x):
        """Defines the standard forward pass for the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The network's output tensor.
        """
        return self.conv(x)
    
    @torch.no_grad()
    def forward_no_grad(self, x):
        """Defines a forward pass without gradient computation.

        Note: This method is redundant as the class's `forward`
        method is also decorated with `@torch.no_grad()`.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The network's output tensor.
        """
        return self.conv(x)

    def load(self, model_name): 
        """Loads pre-trained weights into the model.

        Args:
            model_name (str): The base name of the model file (e.g., "model").
                It first checks the default weights directory or explicit file path,
                then falls back to the configured remote weights directory.
        """
        state_dict = _load_weight_state_dict(model_name, "", self.DEVICE)
        self.load_state_dict(state_dict)
        
class stNet_double(nn.Module):
    """A two-stage refinement network using `stNet` as the base model.

    This model contains two `stNet` instances: one for a "rough" pass
    and one for a "refine" pass. The refinement is iterative.

    Attributes:
        DEVICE (str): The device to run the models on.
        net_rough (stNet): The network for the initial "rough" pass.
        net_refine (stNet): The network for the "refine" passes.
        preprocess (transforms.Compose): Image preprocessing transforms.
    """
    def __init__(self,DEVICE='cuda'):
        """Initializes the stNet_double model.

        Args:
            DEVICE (str, optional): The device to run the models on.
                Defaults to 'cuda'.
        """
        self.DEVICE=DEVICE
        super(stNet_double, self).__init__()
        self.net_rough = stNet(self.DEVICE)#2,1
        self.net_refine = stNet(self.DEVICE)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.net_rough.eval()
        self.net_refine.eval()
        
    def load(self, model_name):
        """Loads pre-trained weights for both rough and refine networks.

        Args:
            model_name (str): The base name for the model files.
        """
        rough_state_dict = _load_weight_state_dict(
            model_name, "_rough", self.DEVICE
        )
        self.net_rough.load_state_dict(rough_state_dict)
        
        refine_state_dict = _load_weight_state_dict(
            model_name, "_ref", self.DEVICE
        )
        self.net_refine.load_state_dict(refine_state_dict)

    @torch.no_grad()   
    def forward(self, x):
        """Defines the standard forward pass for the two-stage network.

        The process is:
        1. y1 = net_rough(x)
        2. y2 = net_refine([img, y1])
        3. y3 = net_refine([img, y2])

        Args:
            x (torch.Tensor): Input tensor (B, 2, H, W), expected to be
                concatenated [image, mask].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - The output of the rough network (y1).
                - The output of the second refinement pass (y3).
        """
        xx=x.split(1,dim=1)
        x_ori_img, x_mask=xx[0],xx[1]
        y1= self.net_rough(x)
        new_x=torch.cat((x_ori_img,y1),dim=1)
        y2= self.net_refine(new_x)
        new_x2=torch.cat((x_ori_img,y2),dim=1)
        y3= self.net_refine(new_x2)
        #print(y3.shape)
        return y1,y3#y1,y3
    
    @torch.no_grad()   
    def center_crop(self,input_c2,rect):
        """Crops a tensor to a specified rectangle.

        Wrapper for `torchvision.transforms.functional.crop`.

        Args:
            input_c2 (torch.Tensor): The input tensor to crop.
            rect (tuple): A tuple (i, j, w, h) defining the crop box.
                Note: `i` is x (width-wise) and `j` is y (height-wise).

        Returns:
            torch.Tensor: The cropped tensor.
        """
        i, j, w, h=rect
        if i<0 or j<0:
            pass
            #print(i, j, h, w)
        input_c2 = TF.crop(input_c2, j, i, h, w)
        #print('stNet_double-center_crop',rect,input_c2.shape)
        return input_c2

    @torch.no_grad()    
    def get_result_s2(self, x):
        """Performs a two-step refinement pass using only the refine network.

        The process is:
        1. y1 = net_refine(x)
        2. y2 = net_refine([img, y1])

        Args:
            x (torch.Tensor): Input tensor (B, 2, H, W), expected to be
                concatenated [image, mask].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - The output of the first refinement pass (y1).
                - The output of the second refinement pass (y2).
        """
        xx=x.split(1,dim=1)
        x_ori_img, x_mask=xx[0],xx[1]
        y1 = self.net_refine.forward(x)
        new_x=torch.cat((x_ori_img,y1),dim=1)
        y2 = self.net_refine.forward(new_x)
        return y1,y2

    @torch.no_grad()    
    def get_result_batch_s2(self, original_img_c1,point_imgs,rect_lst):
        """Processes a batch of cropped image patches using the stage-2 (refine) model.

        This function takes a single large original image, a list of smaller
        mask images (point_imgs), and their corresponding locations (rect_lst).
        It crops the inputs, runs them through the `get_result_s2` (refine-only)
        model in batches, and pastes the results back to full-size.

        Args:
            original_img_c1 (np.ndarray): The full-size original image (H, W, 1).
            point_imgs (list[np.ndarray]): A list of numpy arrays, each
                representing a mask (H, W, 1).
            rect_lst (list[tuple]): A list of (x, y, w, h) rectangles
                corresponding to each mask.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - im_res0 (np.ndarray): Batch of first-pass refinement results (N, H, W, 1).
                - im_res1 (np.ndarray): Batch of second-pass refinement results (N, H, W, 1).
                - im_mask (np.ndarray): Batch of original input masks (N, H, W, 1).
        """
        lst=[]
        mask_lst=[]
        
        t_ori=self.preprocess(original_img_c1).unsqueeze(0).to(self.DEVICE)
        for i in range(len(point_imgs)):
            np_img_c1=point_imgs[i]
            t_mask=self.preprocess(np_img_c1).unsqueeze(0).to(self.DEVICE)
            t_c2=torch.cat((t_ori.detach(),t_mask),dim=1)
            #t_c3=torch.cat((t_ori.detach(),t_ori.detach(),t_mask),dim=1)
            #save_image(self.center_crop(t_c3,rect_lst[i]), 't_c3c_%d.png'%i)
            #save_image(t_c3, 't_c3_%d.png'%i)
            lst.append(self.center_crop(t_c2,rect_lst[i]))
            mask_lst.append(t_mask)
        #print('mask_lst',len(mask_lst),len(point_imgs))
        tmask_lst=torch.cat(mask_lst,dim=0)    
        input=torch.cat(lst,dim=0)
        
        
        #print('input',input.shape,t_ori.shape,t_c2.shape)
        bn=64
        n_mask=len(input)
        nt=(n_mask//bn)+1
        r0_list,r1_list=[],[]
        for i in range(nt):
            input_sub= input[i*bn:min((i+1)*bn,n_mask)]
            t0=time.time()
            res0,res1 = self.get_result_s2(input_sub)
            t1=time.time()
            #print(i,'input',input.shape,t1-t0)
            #save_image(res0, 'res0.png')
            #save_image(res1, 'res1.png')
            #raise
            r0_list.append(res0)
            r1_list.append(res1)
        t_r0=torch.cat(r0_list,dim=0)
        t_r1=torch.cat(r1_list,dim=0)
        #save_image(t_r0, 't_r0.png')
        #save_image(t_r1, 't_r1.png')
        #save_image(pad_back_lst(t_ori,t_r0,rect_lst), 'pad_back_lst.png')
        #raise

        
        im_res0 =(pad_back_lst(t_ori,t_r0,rect_lst).permute(0, 2, 3, 1).detach().cpu().numpy()) * 255#NCHW -> NHWC
        im_res1 =(pad_back_lst(t_ori,t_r1,rect_lst).permute(0, 2, 3, 1).detach().cpu().numpy()) * 255
        im_mask = (tmask_lst.permute(0, 2, 3, 1).detach().cpu().numpy()) * 255
        return np.array(im_res0,np.uint8),np.array(im_res1,np.uint8),np.array(im_mask,np.uint8)
            
    @torch.no_grad()    
    def get_result_batch(self, original_img_c1,point_imgs,rect_lst):
        """Processes a batch of cropped image patches using the full (rough+refine) model.

        This function takes a single large original image, a list of smaller
        mask images (point_imgs), and their corresponding locations (rect_lst).
        It crops the inputs, runs them through the full `forward` (rough+refine)
        model in batches, and pastes the results back to full-size.

        Args:
            original_img_c1 (np.ndarray): The full-size original image (H, W, 1).
            point_imgs (list[np.ndarray]): A list of numpy arrays, each
                representing a mask (H, W, 1).
            rect_lst (list[tuple]): A list of (x, y, w, h) rectangles
                corresponding to each mask.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - im_res0 (np.ndarray): Batch of rough pass results (N, H, W, 1).
                - im_res1 (np.ndarray): Batch of final refinement results (N, H, W, 1).
                - im_mask (np.ndarray): Batch of original input masks (N, H, W, 1).
        """
        lst=[]
        mask_lst=[]
        
        t_ori=self.preprocess(original_img_c1).unsqueeze(0).to(self.DEVICE)
        for i in range(len(point_imgs)):
            np_img_c1=point_imgs[i]
            t_mask=self.preprocess(np_img_c1).unsqueeze(0).to(self.DEVICE)
            t_c2=torch.cat((t_ori.detach(),t_mask),dim=1)
            lst.append(self.center_crop(t_c2,rect_lst[i]))
            mask_lst.append(t_mask)
        tmask_lst=torch.cat(mask_lst,dim=0)    
        input=torch.cat(lst,dim=0)
        #print('get_result_batch start',input.shape,t_ori.shape,t_c2.shape)
        bn=128
        n_mask=len(input)
        nt=(n_mask//bn)+1
        r0_list,r1_list=[],[]
        for i in range(nt):
            input_sub= input[i*bn:min((i+1)*bn,n_mask)]
            t0=time.time()
            res0,res1 = self.forward(input_sub)
            t1=time.time()

            r0_list.append(res0)
            r1_list.append(res1)
        t_r0=torch.cat(r0_list,dim=0)
        t_r1=torch.cat(r1_list,dim=0)

        im_res0 =(pad_back_lst(t_ori,t_r0,rect_lst).permute(0, 2, 3, 1).detach().cpu().numpy()) * 255#NCHW -> NHWC
        im_res1 =(pad_back_lst(t_ori,t_r1,rect_lst).permute(0, 2, 3, 1).detach().cpu().numpy()) * 255
        im_mask = (tmask_lst.permute(0, 2, 3, 1).detach().cpu().numpy()) * 255
        return np.array(im_res0,np.uint8),np.array(im_res1,np.uint8),np.array(im_mask,np.uint8)
