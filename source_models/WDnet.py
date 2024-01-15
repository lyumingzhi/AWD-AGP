import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from inpainting.WDNet.dataloader import dataloader
from inpainting.WDNet.unet_parts import *
from tensorboardX import SummaryWriter
from inpainting.WDNet.vgg import Vgg16

from inpainting.WDNet.WDNet import generator, discriminator

class WDnet(nn.Module):
    def __init__(self, args=None):
        super(WDnet,self).__init__()
        # parameters
        # self.epoch = args.epoch
        # self.batch_size = args.batch_size
        # self.save_dir = args.save_dir
        # self.result_dir = args.result_dir
        # self.dataset = args.dataset
        # self.log_dir = args.log_dir
        # self.gpu_mode = args.gpu_mode
        # self.input_size = args.input_size
        # self.model_name = args.gan_type
        self.z_dim = 62
        self.class_num = 3
        self.sample_num = self.class_num ** 2

        def weight_init(m):
          classname=m.__class__.__name__
          if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data,0.0,0.02)
          elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)
        # networks init
        self.G = generator(3, 3)
        # self.D = discriminator(input_dim=6, output_dim=1)
        # self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        # self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        
        # if self.gpu_mode:
        #     self.G.cuda()
        #     # self.D.cuda()
        #     self.BCE_loss = nn.BCELoss().cuda()
        #     self.l1loss=nn.L1Loss().cuda()
        #     self.loss_mse = nn.MSELoss().cuda()
        # else:
        #     self.BCE_loss = nn.BCELoss()

        self.G.cuda()
        self.G.apply(weight_init)
        # self.D.apply(weight_init)
        #self.load()
        print('---------- Networks architecture -------------')
        #utils.print_network(self.G)
        #utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        # self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        # for i in range(self.class_num):
        #     self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
        #     for j in range(1, self.class_num):
        #         self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        # temp = torch.zeros((self.class_num, 1))
        # for i in range(self.class_num):
        #     temp[i, 0] = i

        # temp_y = torch.zeros((self.sample_num, 1))
        # for i in range(self.class_num):
        #     temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        # self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        # if self.gpu_mode:
        #     self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

        self.G.load_state_dict(torch.load('inpainting/WDNet/WDNet_G.pkl'))
        self.G.eval()
    def forward(self, x, ):
        x = x[:,[2,1,0],...]
        # x = (x - 1) * 2
        G_ ,g_mask, g_alpha, g_w,I_watermark= self.G(x)
        # G_ = G_ / 2 + 1
        G_ = G_[:,[2,1,0],:,:]
        # return G_, g_mask, g_alpha, g_w, I_watermark
        # return G_, g_mask, x * (1-g_alpha) + g_w * g_alpha
        return G_, g_mask, x * (1-g_alpha) + g_w * g_alpha
    
import cv2
import torchvision.transforms as transforms
if __name__ == '__main__':
    model = WDnet()
    # img = cv2.imread('/ntuzfs/mingzhi/inpainting/ensemble/experiment_result/surrogate_Mat_RPN_refined_mask_baseline_perceptual_loss_watermark_rm_HTX_ensemble_iter=30_e=0.05_mask_fix=0.33_sign/Matnet/original_input/9.png')
    img = cv2.imread('/ntuzfs/mingzhi/inpainting/ensemble/1.png')

    img = transforms.ToTensor()(img).unsqueeze(0).cuda()

    output, g_mask, g_alpha, g_w, I_watermark = model(img)
    output = output.detach().cpu().numpy()[0].transpose(1,2,0) * 255
    g_mask = g_mask.detach().cpu().numpy()[0].transpose(1,2,0) * 255
    g_alpha = g_alpha.detach().cpu().numpy()[0].transpose(1,2,0) * 255
    g_w = g_w.detach().cpu().numpy()[0].transpose(1,2,0) * 255
    I_watermark = I_watermark.detach().cpu().numpy()[0].transpose(1,2,0) * 255
    

    print(output, np.max(output), np.min(output))
    assert cv2.imwrite('inpainting/WDNet/WDModel_example_output.jpg', output)
    assert cv2.imwrite('inpainting/WDNet/WDModel_mask_output.jpg', g_mask)
    assert cv2.imwrite('inpainting/WDNet/g_alpha.jpg', g_alpha)
    assert cv2.imwrite('inpainting/WDNet/g_w.jpg', g_w)

    assert cv2.imwrite('inpainting/WDNet/I_watermark.jpg', I_watermark)
    assert cv2.imwrite('inpainting/WDNet/final_output.jpg', img.detach().cpu().numpy()[0].transpose(1,2,0) * 255 * (1 - g_mask/255) + g_w * (g_mask/255))


