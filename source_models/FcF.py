import torch

from inspect import formatargvalues
# from inpainting.RFR_Inpainting.model import RFRNetModel
import torch
import os
import yaml
import time 
import numpy as np
import torchvision
import cv2
import inpainting.FcF_Inpainting.dnnlib as dnnlib
from icecream import ic
import pickle

# import inpainting.FcF_Inpainting.legacy as legacy
import sys

from  inpainting.FcF_Inpainting.training.networks import  Generator
class FcFnetAPI(torch.nn.Module):
    def __init__(self,dataset,device=None,class_idx=None, opt=None):
        super(FcFnetAPI,self).__init__()
        self.opt=opt

        config=self.load_config()
        
        self.module=Generator(**config)
        

        # sys.path.append('/home1/mingzhi/inpainting/FcF_Inpainting/')
        # network_pkl='/home1/mingzhi/inpainting/FcF_Inpainting/places.pkl'
        # print('Loading networks from "%s"...' % network_pkl)

        # # with open(network_pkl,'rb') as f:
        # #     data=pickle.load(f)
        #     # print(data)
        #     # exit()
        # with dnnlib.util.open_url(network_pkl) as f:
        #     G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
        
        # sys.path.remove('/home1/mingzhi/inpainting/FcF_Inpainting/')

        checkpoint=torch.load('inpainting/FcF_Inpainting/G.pt')
        self.module.load_state_dict(checkpoint)
        self.module.eval().cuda()
        ic(self.module.encoder.b256.img_channels)
        # self.module.G.eval()

        # print(self.module.c_dim,device)
        # exit()
        self.label = torch.zeros([1, self.module.c_dim]).to(device)
        if self.module.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            self.label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')
            
    def forward(self,x,masks,keepFeat=False):
        masks=1-masks
        
        # gt_images, masks = self.__cuda__(*items)
        x=x*2-1
        original_x=x.clone()
        x=x[:,[2,1,0],...]
        x=torchvision.transforms.Compose([ torchvision.transforms.Resize((256,256))])(x)
        masks=torchvision.transforms.Compose([ torchvision.transforms.Resize((256,256))])(masks)
        masked_images = x * masks
        # cv2.imwrite('original_input.jpg',masked_images[0].detach().cpu().numpy().transpose(1,2,0)*255)
        # exit()
        # masks = torch.cat([masks]*3, dim = 1)
        # print(masks.size())
        # exit(0)

        truncation_psi=0.1  
        if keepFeat==False:
            # print(masked_images.size(),masks.size())
            # print(torch.cat([0.5 - (1-masks), masked_images]).size())
            # exit()
            fake_B = self.module(img=torch.cat([0.5 - (1-masks), masked_images], dim=1), c=self.label, truncation_psi=truncation_psi, noise_mode='const',keepFeat=keepFeat)
        else:
            fake_B, FeatList = self.module(img=torch.cat([0.5 - (1-masks), masked_images], dim=1), c=self.label, truncation_psi=truncation_psi, noise_mode='const',keepFeat=keepFeat)
            # fake_B, mask, FeatList = self.module(masked_images, 1-masks,keepFeat=keepFeat)
            # print(masks)
            # exit()
        if self.opt.wo_mask:
            comp_B=fake_B
        else:
            comp_B = fake_B * (1 - masks) + masked_images*masks
        comp_B=comp_B[:,[2,1,0],...]

        comp_B=comp_B/2+0.5

        # print(torch.max(torch.abs(comp_B-x)))
        # assert cv2.imwrite('inpainting/ensemble/inout_diff.jpg',(torch.abs(comp_B-x))[0].detach().cpu().numpy().transpose(1,2,0)*255)
        # exit()
        
        if keepFeat==False:
            return comp_B
        else:
            return comp_B, FeatList
    def load_config(self):
        cfg='auto'
        gpus=1
        cfg_specs = {
            'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
            'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
            'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
            'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
            'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
            'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        }

        assert cfg in cfg_specs
        spec = dnnlib.EasyDict(cfg_specs[cfg])

        if cfg == 'auto':
            # desc += f'{gpus:d}'
            spec.ref_gpus = gpus
            res =  256
            spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
            spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
            spec.fmaps = 1 if res >= 512 else 0.5
            spec.lrate = 0.001 if res >= 1024 else 0.001
            spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
            spec.ema = spec.mb * 10 / 32
        G_kwargs = dnnlib.EasyDict( z_dim=512, w_dim=512, encoder_kwargs=dnnlib.EasyDict(block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs = dnnlib.EasyDict()), mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
        G_kwargs.synthesis_kwargs.channel_base = G_kwargs.encoder_kwargs.channel_base = int(spec.fmaps * 32768)
        G_kwargs.synthesis_kwargs.channel_max =  G_kwargs.encoder_kwargs.channel_max  = 512
        G_kwargs.mapping_kwargs.num_layers = spec.map
        G_kwargs.synthesis_kwargs.num_fp16_res =G_kwargs.encoder_kwargs.num_fp16_res  = 4 # enable mixed-precision training
        G_kwargs.synthesis_kwargs.conv_clamp = G_kwargs.encoder_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
        G_kwargs.encoder_kwargs.epilogue_kwargs.mbstd_group_size =  spec.mbstd
        G_kwargs.img_resolution=256
        G_kwargs.w_dim=512
        G_kwargs.img_channels=3
        G_kwargs.c_dim=0

        return G_kwargs


# class RFR_Config(dict):
#     def __init__(self, config_path):
#         with open(config_path, 'r') as f:
#             self._yaml = f.read()
#             self._dict = yaml.safe_load(self._yaml)
#             self._dict['Conifg_PATH'] = os.path.dirname(config_path)

#         # self.parse()
#         # print(self)
#     def __getattr__(self, name):
#         if self._dict.get(name) is not None:
#             return self._dict[name]

#         # if DEFAULT_CONFIG.get(name) is not None:
#         #     return DEFAULT_CONFIG[name]

#         return None

#     def print(self):
#         print('Model configurations:')
#         print('---------------------------------')
#         print(self._yaml)
#         print('')
#         print('---------------------------------')
#         print('')

   

if __name__=='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import cv2
    import torchvision.transforms as transforms
    model = FcFnetAPI('place2')

    # model.initialize_model(args.model_path, False)
    # model.cuda()
    # dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True, training=False))
    # model.test(dataloader, args.result_save_path)
    img=cv2.imread('inpainting/FcF_Inpainting/datasets/demo/places2/200_Places365_val_00011651.png') 
    mask=cv2.imread('inpainting/FcF_Inpainting/datasets/demo/places2/200_Places365_val_00011651_mask.png')[:,:,0]
    # print(mask)
    mask=(1-(mask > 0.1).astype(np.uint8))*255

    # print(img)
    img=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])(img.copy()).cuda().unsqueeze(0)
    mask=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])(mask.copy()).cuda().unsqueeze(0)
    # print(img.size(),mask)
    output=model(img,mask)
    assert cv2.imwrite('inpainting/FcF_Inpainting/example_output.jpg',output[0].detach().cpu().numpy().transpose(1,2,0)*255)
