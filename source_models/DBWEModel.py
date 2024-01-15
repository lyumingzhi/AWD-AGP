from __future__ import print_function, absolute_import

import argparse
import torch
import os
from math import log10
import cv2
import numpy as np

torch.backends.cudnn.benchmark = True

# import official_datasets as datasets
from inpainting.DBWEModel.options import Options
import torch.nn.functional as F
# import pytorch_ssim
# from evaluation import compute_IoU, FScore, AverageMeter, compute_RMSE, normPRED
# from skimage.measure import compare_ssim as ssim
import time
import inpainting.DBWEModel.scripts.models as archs
import torch.nn as nn 
def is_dic(x):
    return type(x) == type([])

class DBWRnet(nn.Module):
    def __init__(self, args=None):
        super(DBWRnet,self).__init__()
        parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
        self.args = parser.parse_args()
        # Machine = models.__dict__[args.models](datasets=data_loaders, args=args)

        # create model
        print("==> creating model ", self.args.arch)
        # self.model = archs.__dict__[self.args.arch]()
        self.model = archs.__dict__['vvv4n']().cuda()
        print("==> creating model [Finish]")

        self.model.cuda()
        

        resume_path = '/ntuzfs/mingzhi/inpainting/DBWEModel/27kpng_model_best.pth.tar'
        print("=> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)
        # print('current ckpt', current_checkpoint['arch'])
        # exit()
        if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
            current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module
        
        self.model.load_state_dict(current_checkpoint['state_dict'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, current_checkpoint['epoch']))
        

    def forward(self, x, ):
        # self.model.eval()

        x = x[:,[2,1,0],...]
        imoutput,immask_all,imwatermark = self.model(x)
        # imoutput = imoutput[0] if is_dic(imoutput) else imoutput
        imoutput,imfinal,imwatermark = imoutput[1]*immask_all + x*(1-immask_all),imoutput[0]*immask_all + x*(1-immask_all),imwatermark*immask_all
            
        immask = immask_all[0]
        
        

        # imfinal = imoutput*immask + x*(1-immask)

        # imfinal = (imfinal + 1) / 2

        imoutput = imoutput[:,[2,1,0],:,:]
        imfinal = imfinal[:,[2,1,0],:,:]
        # return G_, g_mask, g_alpha, g_w, I_watermark
        return imoutput, immask, imfinal
    
import cv2
import torchvision.transforms as transforms
if __name__ == '__main__':
    model = DBWEModel()
    # img = cv2.imread('/ntuzfs/mingzhi/inpainting/ensemble/experiment_result/surrogate_Mat_RPN_refined_mask_baseline_perceptual_loss_watermark_rm_HTX_ensemble_iter=30_e=0.05_mask_fix=0.33_sign/Matnet/original_input/9.png')
    img = cv2.imread('/ntuzfs/mingzhi/inpainting/ensemble/1.png')

    img = transforms.ToTensor()(img).unsqueeze(0).cuda()

    imoutput, immask, imfinal = model(img)
    imoutput = imoutput.detach().cpu().numpy()[0].transpose(1,2,0) * 255
   
    immask = immask.detach().cpu().numpy()[0] * 255
    imfinal = imfinal.detach().cpu().numpy()[0].transpose(1,2,0) * 255
    # g_w = g_w.detach().cpu().numpy()[0].transpose(1,2,0) * 255
    # I_watermark = I_watermark.detach().cpu().numpy()[0].transpose(1,2,0) * 255
    

    # print(output, np.max(output), np.min(output))
    assert cv2.imwrite('inpainting/DBWEModel/DBWEModel_example_output.jpg', imoutput)
    assert cv2.imwrite('inpainting/DBWEModel/DBWEModel_mask_output.jpg', immask)
    assert cv2.imwrite('inpainting/DBWEModel/DBWEModel_final_output.jpg', imfinal)
    # assert cv2.imwrite('inpainting/WDNet/g_w.jpg', g_w)

    # assert cv2.imwrite('inpainting/WDNet/I_watermark.jpg', I_watermark)
    # assert cv2.imwrite('inpainting/WDNet/final_output.jpg', img.detach().cpu().numpy()[0].transpose(1,2,0) * 255 * (g_mask/255) + g_w * (1-g_mask/255))


