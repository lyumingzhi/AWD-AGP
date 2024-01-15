from inpainting.inpainting_gmcnn.pytorch.model.net import InpaintingModel_GMCNN
from inpainting.inpainting_gmcnn.pytorch.options.read_config import GMCNNConfig
from inpainting.inpainting_gmcnn.pytorch.util.utils import generate_rect_mask, generate_stroke_mask, getLatest
import os
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
class GMCNNAPI(torch.nn.Module):
    def __init__(self,dataset='celeba',opt=None):
        super(GMCNNAPI,self).__init__()
        self.dataset=dataset
        self.opt=opt
        config=self.load_Config()
        if dataset!='places2':
            self.module= InpaintingModel_GMCNN(in_channels=4, opt=config,dataset=dataset)
        else:
            self.module= InpaintingModel_GMCNN(in_channels=5, opt=config,dataset=dataset)

        # print('inference',self.module.single_inference)
        # for att in vars(self.module):
        #     print(att)
        # exit()
        self.config=config
        if config.load_model_dir != '' :
            if dataset=='celeba':
                print('Loading pretrained model from {}'.format(config.load_model_dir))
                self.module.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
                print('Loading done.')
            if dataset=='places2':
                print('Loading pretrained model from {}'.format('inpainting/inpainting_gmcnn/pytorch/chkpts/places2_rect/'))
                self.module.load_networks(getLatest(os.path.join('inpainting/inpainting_gmcnn/pytorch/chkpts/places2_rect', '*.pth')))
                print('Loading done.')
            # exit()
    #     print('inference',self.module.single_inference)
    #     exit()
    def forward(self,x,mask,keepFeat=False):
        # print('api',keepFeat)
        # exit()
        original_input=x
        x=x[:,[2,1,0],...]
        # print(x)
        # exit()
        if self.dataset=='celeba':
            result=self.module.single_inference(x, mask)
        elif self.dataset=='places2':
            if keepFeat==False:
                result=self.module.single_inference_fromTF(x, mask,keepFeat=keepFeat)
            else:
                
                result, FeatList=self.module.single_inference_fromTF(x, mask,keepFeat=keepFeat)
        
        # print(len(FeatList))
        # exit()
        result=result[:,[2,1,0],...]
        if self.opt.wo_mask:
            result=result*mask+original_input*(1-mask)
        
        if keepFeat==False:
            return result
        else:
            
            return result, FeatList
    def evaluate(self,*args,**kwargs):
        return self.module.evaluate(*args,**kwargs)

    def load_Config(self):
        config=GMCNNConfig('inpainting/inpainting_gmcnn/pytorch/options/config.yaml')
        config.data_file='./imgs/celebahq_256x256/'
        return config

def generate_mask_rect(im_shapes, mask_shapes, rand=True):
    mask = np.zeros((im_shapes[0], im_shapes[1])).astype(np.float32)
    if rand:
        of0 = np.random.randint(0, im_shapes[0]-mask_shapes[0])
        of1 = np.random.randint(0, im_shapes[1]-mask_shapes[1])
    else:
        of0 = (im_shapes[0]-mask_shapes[0])//2
        of1 = (im_shapes[1]-mask_shapes[1])//2
    mask[of0:of0+mask_shapes[0], of1:of1+mask_shapes[1]] = 1
    mask = np.expand_dims(mask, axis=2)
    return mask
if __name__=='__main__':
    

    # config=Config('/home1/mingzhi/inpainting/inpainting_gmcnn/pytorch/options/config.yaml')
    # config.data_file='./imgs/celebahq_256x256/'
    img=cv2.imread('/home1/mingzhi/inpainting/inpainting_gmcnn/tensorflow/imgs/paris-streetview_256x256/001.png')[...,::-1]
    cv2.imwrite('inpainting/ensemble/img_input_example.jpg',img)
    # mask=cv2.imread('/home1/mingzhi/inpainting/edge_connect/examples/celeba/masks/celeba_01.png')[:,:,0]
    mask=np.zeros(img.shape)
    mask[mask.shape[0]//4:3*mask.shape[0]//4,mask.shape[1]//4:3*mask.shape[1]//4,:]=1
    mask = generate_mask_rect([256,256,3], [128,128], False)

    grid=4
    h,w=256,256
    # img=img[:h // grid * grid, :w // grid * grid, :]
    # mask = mask[:h // grid * grid, :w // grid * grid, :]

    # print(mask)
    # mask=(1-(mask > 0.1).astype(np.uint8))*255

    # print(img)
    # print(img)
    img=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])(img.copy()).cuda().unsqueeze(0)
    mask=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])(mask.copy()).cuda().unsqueeze(0)
    # print(img.size(),mask)
    model=GMCNNAPI('places2')
    
    
    # assert cv2.imwrite('inpainting/ensemble/place_example.jpg',img[0].detach().cpu().numpy().transpose(1,2,0)*255)
    # exit()
    # print(img)
    # exit(0)
    output=model(img,mask)

    assert cv2.imwrite('/home1/mingzhi/inpainting/ensemble/gmcnn_place2.jpg',output[0].detach().cpu().numpy()[::-1,...].transpose(1,2,0)*255)
    # print('hhh',config.test_num)
