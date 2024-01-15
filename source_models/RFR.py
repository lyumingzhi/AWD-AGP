from inspect import formatargvalues
from inpainting.RFR_Inpainting.model import RFRNetModel
import torch
import os
import yaml
import time 
import numpy as np
import torchvision
import cv2
class RFRNetModelAPI(torch.nn.Module):
    def __init__(self,dataset,opt):
        super(RFRNetModelAPI,self).__init__()
        config=self.load_config()
        self.module=RFRNetModel()
        self.opt=opt
        
        if dataset=='celeba':
            self.module.initialize_model(config.model_path, False)
        elif dataset=='places2':
            self.module.initialize_model('/home1/mingzhi/inpainting/RFR_Inpainting/RFR_Net_pretrained/checkpoint_paris.pth',False)
        self.module.cuda()
        self.module.G.eval()
    def forward(self,x,masks,keepFeat=False):
        masks=1-masks
        # gt_images, masks = self.__cuda__(*items)
        original_x=x.clone()
        x=x[:,[2,1,0],...]
        x=torchvision.transforms.Compose([ torchvision.transforms.Resize((256,256))])(x)
        masks=torchvision.transforms.Compose([ torchvision.transforms.Resize((256,256))])(masks)
        masked_images = x * masks
        # cv2.imwrite('original_input.jpg',masked_images[0].detach().cpu().numpy().transpose(1,2,0)*255)
        # exit()
        masks = torch.cat([masks]*3, dim = 1)
        # print(masks.size())
        # exit(0)
        if keepFeat==False:
            fake_B, mask = self.module.G(masked_images, masks,keepFeat=keepFeat)
        else:
            fake_B, mask, FeatList = self.module.G(masked_images, masks,keepFeat=keepFeat)

        if self.opt.wo_mask:
            comp_B=fake_B
        else:
            comp_B = fake_B * (   1 - masks) + x * masks
        
        comp_B=comp_B[:,[2,1,0],...]

        # print(torch.max(torch.abs(comp_B-x)))
        # assert cv2.imwrite('inpainting/ensemble/inout_diff.jpg',(torch.abs(comp_B-x))[0].detach().cpu().numpy().transpose(1,2,0)*255)
        # exit()
        
        if keepFeat==False:
            return comp_B
        else:
            return comp_B, FeatList
    def load_config(self):
        config=RFR_Config('inpainting/RFR_Inpainting/RFR_Net_pretrained/celeba.yaml')
        return config


class RFR_Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['Conifg_PATH'] = os.path.dirname(config_path)

        # self.parse()
        # print(self)
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        # if DEFAULT_CONFIG.get(name) is not None:
        #     return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')

   

if __name__=='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import cv2
    import torchvision.transforms as transforms
    model = RFRNetModelAPI()

    # model.initialize_model(args.model_path, False)
    # model.cuda()
    # dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True, training=False))
    # model.test(dataloader, args.result_save_path)
    img=cv2.imread('/home1/mingzhi/inpainting/edge_connect/examples/celeba/images/celeba_01.png')[...,::-1]
    mask=cv2.imread('/home1/mingzhi/inpainting/edge_connect/examples/celeba/masks/celeba_01.png')[:,:,0]
    # print(mask)
    mask=(1-(mask > 0.1).astype(np.uint8))*255

    # print(img)
    img=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])(img.copy()).cuda().unsqueeze(0)
    mask=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])(mask.copy()).cuda().unsqueeze(0)
    # print(img.size(),mask)
    output=model(img,mask)
    assert cv2.imwrite('/home1/mingzhi/inpainting/RFR_Inpainting/example_output.jpg',output[0].detach().cpu().numpy().transpose(1,2,0)*255)
