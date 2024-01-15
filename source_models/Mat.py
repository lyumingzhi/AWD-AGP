from pydoc import resolve
from inpainting.MAT.networks.mat import Generator
import torch
import torchvision
import inpainting.MAT.dnnlib as dnnlib
import numpy as np
import inpainting.MAT.legacy as legacy

class MatAPI(torch.nn.Module):
    def __init__(self,opt,device='cuda'):
        super(MatAPI,self).__init__()
        self.opt=opt
        self.resolution=512
        self.device=device
        self.module=Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=self.resolution, img_channels=3).to(device).eval().requires_grad_(False)
        # with dnnlib.util.open_url('/home1/mingzhi/inpainting/MAT/pretrained/Places_512_FullData.pkl') as f:
        #     G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore

        # self.copy_params_and_buffers(G_saved, self.module, require_all=True)

        self.module.load_state_dict(torch.load('inpainting/MAT/pretrained/Places_512_FullData_real.pkl'))
        self.module.eval()
    def forward(self,x,masks,keepFeat=False):
        # masks=1-masks
                
        
        # print('mask', masks)
        # gt_images, masks = self.__cuda__(*items)
        original_x=x.clone()
        x=x[:,[2,1,0],...]
        x=torchvision.transforms.Compose([ torchvision.transforms.Resize((self.resolution,self.resolution))])(x)
        masks=torchvision.transforms.Compose([ torchvision.transforms.Resize((self.resolution,self.resolution))])(masks)

        masks=(masks>0.1).to(self.device)*1.0
        masks=1-masks
        masked_images = x * masks
        # cv2.imwrite('inpainting/ensemble/original_input.jpg',masked_images[0].detach().cpu().numpy().transpose(1,2,0)*255)
        # exit()
        # masks = torch.cat([masks]*3, dim = 1)
        # print(masks.size())
        # exit(0)

        masked_images=masked_images*2-1

        label = torch.zeros([1, self.module.c_dim], device=self.device)
        z = torch.from_numpy(np.random.randn(1, self.module.z_dim)).to(self.device)
        truncation_psi=1.0
        noise_mode='const'

        if keepFeat==False:
            fake_B = self.module(masked_images, masks,z,label,truncation_psi=truncation_psi, noise_mode=noise_mode )
        else:
            fake_B, FeatList = self.module(masked_images, masks, z,label,truncation_psi=truncation_psi, noise_mode=noise_mode,keepFeat=keepFeat)

        # if self.opt.wo_mask:
        #     comp_B=fake_B
        # else:
        #     comp_B = fake_B * (   1 - masks) + x * masks
        
        comp_B=fake_B/2+0.5

        # print(torch.max(torch.abs(comp_B-x)))
        # assert cv2.imwrite('inpainting/ensemble/inout_diff.jpg',(torch.abs(comp_B-x))[0].detach().cpu().numpy().transpose(1,2,0)*255)
        # exit()
        comp_B=comp_B[:,[2,1,0],:,:]
        if keepFeat==False:
            return comp_B
        else:
            return comp_B, FeatList
    def copy_params_and_buffers(self,src_module, dst_module, require_all=False):
        assert isinstance(src_module, torch.nn.Module)
        assert isinstance(dst_module, torch.nn.Module)
        src_tensors = {name: tensor for name, tensor in self.named_params_and_buffers(src_module)}
        for name, tensor in self.named_params_and_buffers(dst_module):
            assert (name in src_tensors) or (not require_all)
            if name in src_tensors:
                tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
    def named_params_and_buffers(self,module):
        assert isinstance(module, torch.nn.Module)
        return list(module.named_parameters()) + list(module.named_buffers())


    
    
def generate_rect_mask( im_size, mask_size, margin=8, rand_mask=True):
    if mask_size==None:
        mask_size=(np.random.randint(int(0.2*im_size[0]),int(0.5*im_size[0])),int(0.2*im_size[1]),int(0.5*im_size[1]))
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)
    if rand_mask:
        sz0, sz1 = mask_size[0], mask_size[1]
        # print(margin, im_size[0] - sz0 - margin)
        # exit()
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
    else:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = (im_size[0] - sz0) // 2
        of1 = (im_size[1] - sz1) // 2
    mask[of0:of0+sz0, of1:of1+sz1] = 1
    mask = np.expand_dims(mask, axis=0)
    
    rect = torch.from_numpy(np.array([[of0, sz0, of1, sz1]], dtype=int))
    return mask, rect
if __name__=='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import cv2
    import torchvision.transforms as transforms

    model = MatAPI(opt=None)

    # model.initialize_model(args.model_path, False)
    # model.cuda()
    # dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True, training=False))
    # model.test(dataloader, args.result_save_path)
    img=cv2.imread('/home1/mingzhi/inpainting/ensemble/experiment_result/surrogate_gradient_loss_inter_grad_inpaintingModel_places2_sign_target_attack/Matnet/attacked_input/199.png')[...,::-1]
    mask=cv2.imread('/home1/mingzhi/inpainting/MAT/test_sets/Places/masks/mask1.png')[:,:,0]
    mask=np.zeros((512,512,3))
    # mask[mask.shape[0]//2:3*mask.shape[0]//4,mask.shape[1]//4:3*mask.shape[1]//4,:]=1
    mask ,_= generate_rect_mask([512,512,3], [128,128], False)
    
    print(mask.shape)
    mask=(1-(mask.transpose(1,2,0) > 0.1).astype(np.uint8))*255

    # print(img)
    img=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])(img.copy()).cuda().unsqueeze(0)
    mask=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512))])(mask.copy()).cuda().unsqueeze(0)
    print(img.size(),mask.size())
    output=model(img,mask)
    assert cv2.imwrite('/home1/mingzhi/inpainting/MAT/example_output.jpg',output[0].detach().cpu().numpy().transpose(1,2,0)*255)
