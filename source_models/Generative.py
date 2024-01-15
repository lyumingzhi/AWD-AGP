import torch
from inpainting.generative_inpainting_pytorch.trainer import Trainer
import imageio
import numpy as np
import cv2
from inpainting.generative_inpainting_pytorch.utils.tools import get_config
class Generative(torch.nn.Module):
    def __init__(self,model_path='inpainting/generative_inpainting_pytorch/torch_model.p',device='cuda',config='inpainting/generative_inpainting_pytorch/configs/config.yaml',dataset='celeba', opt=None):

        super(Generative,self).__init__()
        config = get_config(config)
        if dataset=='places2':
            model_path='inpainting/generative_inpainting_pytorch/chkpt/places2.pth'
            # print(model_path)
            self.module=Trainer(config,FromTF=True)
        elif dataset=='celeba':
            self.module=Trainer(config,FromTF=False)

        self.opt=opt

        self.dataset=dataset
        # print(self.load_weights(model_path,device ).keys())
        # print(self.load_weights(model_path,device )['coarse_generator.conv1.conv.bias'])
        # exit()
        self.module.netG.load_state_dict(self.load_weights(model_path,device ), strict=True)
        # print(self.module.netG.state_dict()['coarse_generator.conv1.conv.bias'])
        # exit()
        self.module.eval()
    def forward(self,x,mask,keepFeat=False):
        # h, w= x.size()[2],x.size()[3]
        # grid = 8
        # x = x.clone()[:,:,:h//grid*grid, :w//grid*grid]
        # mask = mask.clone()[:,:,:h//grid*grid, :w//grid*grid]


        if self.dataset=='celeba':
            x=x[:,[2,1,0],...]
        elif self.dataset=='places2':
            x=x[:,[2,1,0],...]
        original_x=x
        x=x*2-1
        x=x*(1-mask)
        
        if keepFeat==False:
            _,result,_=self.module.netG(x, mask,keepFeat=keepFeat)
        else:
            _,result,_,FeatList=self.module.netG(x, mask,keepFeat=keepFeat)
        result=result[:,[2,1,0],...]
        result=result/2+0.5

        if not self.opt.wo_mask:
            result=result*mask+original_x[:,[2,1,0],...]*(1-mask)

        if keepFeat==False:
            return result
        else:
            return result, FeatList
    def load_weights(self,path, device):
        model_weights = torch.load(path)
        return {
            k: v.to(device)
            for k, v in model_weights.items()
        }
def upcast(x):
    return np.clip((x + 1) * 127.5 , 0, 255).astype(np.uint8)
if __name__=='__main__':
    
    trainer = Generative(dataset='places2')
    trainer.eval()
    # trainer.load_state_dict(load_weights(args.model_path, device), strict=False)
    # trainer.eval()

    # image = cv2.imread('inpainting/generative_inpainting_pytorch/examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png')
    image=cv2.imread('/home1/mingzhi/inpainting/generative_inpainting/examples/places2/case6_input.png')
    # image=cv2.resize(image,(256,256))
    
    mask = cv2.imread('/home1/mingzhi/inpainting/generative_inpainting/examples/places2/case6_mask.png')
    # mask=cv2.resize(mask,(256,256))
    assert image.shape == mask.shape

    # h, w, _ = image.shape
    # grid = 8
    # image = image[:h//grid*grid, :w//grid*grid, :]
    # mask = mask[:h//grid*grid, :w//grid*grid, :]

    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).cuda()
    mask = (torch.FloatTensor(mask[:, :, 0]) / 255).unsqueeze(0).unsqueeze(0).cuda()
    

    # x = (image / 127.5 - 1) * (1 - mask).cuda()
    x=image.cuda()/255
    with torch.no_grad():
    
        result= trainer(x, mask)

    imageio.imwrite('inpainting/generative_inpainting_pytorch/output_example.jpg', upcast(result[0].permute(1, 2, 0).detach().cpu().numpy()))