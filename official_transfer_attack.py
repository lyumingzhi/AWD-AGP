# from inpainting.RFR_Inpainting.run import attack
from sympy import O
# from inpainting.RFR_Inpainting.run import attack
from inpainting.ensemble.Attack import Attacker
import argparse
import torch
from inpainting.ensemble.load_models import load_models
import cv2

import kornia as K
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader

from inpainting.ensemble.official_dataset import AttackDataset
from inpainting.ensemble.official_dataset import TransferAttackDataset
from pathlib import Path

import json

import os

# run sh
# CUDA_VISIBLE_DEVICES=7 python -m inpainting.ensemble.official_transfer_attack --dataset places2 --lossType perceptual_loss --InputImg_dir /home1/mingzhi/inpainting/Watermarking  --output_dir transfer_Mat_RPN_refined_mask_baseline_perceptual_loss_watermark_rm_HTX_ensemble_iter=30_e=0.05_mask_fix=0.33_sign --experiment_dir /home1/mingzhi/inpainting/ensemble/HTX_experiment/surrogate_Mat_RPN_refined_mask_baseline_perceptual_loss_watermark_rm_HTX_ensemble_iter=30_e=0.05_mask_fix=0.33_sign --target_model Matnet --algorithm optimal_mask_search --get_logo --attach_logo

# get edges of images
def get_mask_edge(mask):
    edges=K.filters.canny(mask,sigma=(2,2),low_threshold=0.05,high_threshold=0.1,kernel_size=(21,21))[1]
    return edges

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target_model',type=str,required=False,help='the surrogant model')    
    parser.add_argument('--attackType',action='append',default=[],required=False,help='the attack method')
    parser.add_argument('--algorithm',action='append',default=[],type=str,help='algorithmf for attack')
    parser.add_argument('--lossType',action='append',default=[],help='loss type for attack')
    parser.add_argument('--random_mask',action='store_true',help='whehter use random mask when attack')
    parser.add_argument('--sign',action='store_true',help='whehter use sign of gradient when doing attack')
    parser.add_argument('--InputImg_dir',type=str,required=False,help='the path to input images')
    parser.add_argument('--target_img',type=str,default='',help='target image for target attack')
    parser.add_argument('--target_attack',action='store_true',help='whether to use target attack')
    parser.add_argument('--output_dir',type=str,required=True,help='path to output images')
    parser.add_argument('--attack_specific_region',action='store_true',help='whether to just attack specific region')
    parser.add_argument('--dataset',type=str,default='celeba',help='which dataset the target models are trained on')
    parser.add_argument('--attack_type',action='append',default=[],help='which attack objective is used')
    parser.add_argument('--experiment_dir',type=str,default='',help='experiment name of the adversarial generation experiment')
    parser.add_argument('--wo_mask',action='store_true',default=False,help='whether to use raw output of inpainting models')
    parser.add_argument('--get_logo', action = 'store_true', default=False, help='whether to get a example logo')
    parser.add_argument('--attach_logo', action = 'store_true', default=False, help='whether to attach logo to the dataset image')
    parser.add_argument('--watermark_alpha', type = float, default=0.5, help ='the alpha to embed watermark to the image')
    parser.add_argument('--relative_mask_size', type = float, default=0.33, help ='the mask size of the watermark to the image')
    opt=parser.parse_args()



    
    
   

    RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet=load_models(opt)
    # RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet, WDModel, DBWEModel, SLBRModel=load_models(opt)
    # del RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet

    # modelsToTest={'RFRnet':RFRnet,'GMCNNnet':GMCNNnet,'EdgeConnet':EdgeConnet,'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,}
    if opt.target_model=='Matnet':
        modelsToTest={'Matnet':Matnet}
    elif opt.target_model=='RFRnet':
        modelsToTest={'RFRnet':RFRnet}
    elif opt.target_model=='GMCNNnet':
        modelsToTest={'GMCNNnet':GMCNNnet}
    elif opt.target_model=='EdgeConnet':
        modelsToTest={'EdgeConnet':EdgeConnet}
    elif opt.target_model=='Crfillnet':
        modelsToTest={'Crfillnet':Crfillnet}
    elif opt.target_model=='Gennet':
        modelsToTest={'Gennet':Gennet}
    elif opt.target_model=='FcFnet':
        modelsToTest={'FcFnet':FcFnet}
    elif opt.target_model == 'WDModel':
        modelsToTest={'WDModel':WDModel}
    elif opt.target_model == 'DBWEModel':
        modelsToTest={'DBWEModel':DBWEModel}
    elif opt.target_model == 'SLBRModel':
        modelsToTest={'SLBRModel':SLBRModel}
        
    # models={'GMCNNnet':GMCNNnet}
    # models={'RFRnet':RFRnet}
    # models={'EdgeConnet':EdgeConnet}
    # models={'Crfillnet':Crfillnet}
    # models={'Gennet':Gennet}
    print('after loading model')


    randomBlur=transforms.GaussianBlur(57,(1.0,3.0))
    
    for name_to_test in modelsToTest:
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name_to_test,'result/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name_to_test,'annot_result/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name_to_test,'attacked_input/')).mkdir(parents=True,exist_ok=True)
        if 'random_mask' in opt.algorithm:
            Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name_to_test,'random_mask/')).mkdir(parents=True,exist_ok=True)
        dataset=TransferAttackDataset(opt,opt.InputImg_dir,opt.experiment_dir,name_to_test, if_sort=True)
        if dataset==None:
            continue
        dataloader=DataLoader(dataset,batch_size=1,shuffle=False)

        models={name_to_test:modelsToTest[name_to_test]}

        attacker=Attacker(opt,{name_to_test:models[name_to_test]})

        rect_list={}
        for name in modelsToTest.keys():
            rect_list[name]=[]
        rect_list['mask_size']=[]
        preprocess_num=0

        print('start attack')
        
        print('len dataloader',len(dataloader))
        for index,(img, adv_img, mask,rect,target_patch, logo) in enumerate(iter(dataloader)):
            # if index<19:
            #     preprocess_num+=1
            #     continue
            # if index<=405:
            #     preprocess_num+=1
            #     continue
            img=img.cuda()
            adv_img=adv_img.cuda()
          
            mask=(mask.cuda()>0.9)*1.0
            if 'random_mask' in opt.algorithm:
                random_mask=(random_mask.cuda()>0.9)*1.0

            if 'random_blur' in opt.algorithm:
                mask=(randomBlur(mask)>0.5)*1.0
            if opt.target_attack and opt.target_img!='':
                target_patch=target_patch.cuda()
            # print(rect.size())
            # exit()

            print('start to generate original output')
            # exit()
            original_output={}
            original_FeatList={}
            attacked_output={}
            with torch.no_grad():
                # for name in models.keys():
                print(name_to_test,'to play')
                
                # generate the original outputs
                if 'random_mask' in opt.algorithm:
                    if name_to_test in ['WDModel', 'DBWEModel', 'SLBRModel']:
                        original_output[name_to_test]=models[name_to_test](img)
                    else:
                        original_output[name_to_test]=models[name_to_test](img,random_mask)
                else: 
                    if name_to_test in ['WDModel', 'DBWEModel', 'SLBRModel']:
                        original_output[name_to_test]=models[name_to_test](img)
                    else:    
                        original_output[name_to_test]=models[name_to_test](img,mask)

                if name_to_test in ['WDModel', 'DBWEModel', 'SLBRModel']:
                    attacked_output[name_to_test]=models[name_to_test](adv_img)
                else:
                    attacked_output[name_to_test]=models[name_to_test](adv_img,mask)
             
          
            
            size_of_mask=torch.sum(mask>0).item()
            edge=get_mask_edge(mask)
            assert cv2.imwrite('inpainting/ensemble/edgeExample.jpg',edge[0].detach().cpu().numpy().transpose(1,2,0))

            # save the original and attack output
            for i in range(attacked_output[list(attacked_output.keys())[0]].size()[0]):
                
                for name in attacked_output.keys():
                    
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name_to_test,'result/attacked_'+name+'_'+str(preprocess_num)+'.png'),attacked_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/', opt.output_dir,name_to_test,'result/original_'+name+'_'+str(preprocess_num)+'.png'),original_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
       
                    if mask!=None:
                        attacked_output[name]=torch.clamp(attacked_output[name][i]+transforms.Resize((attacked_output[name].size()[-1],attacked_output[name].size()[-1]))(edge),min=0,max=1)
               

                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/', opt.output_dir,name_to_test,'annot_result/attacked_'+name+'_'+str(preprocess_num)+'.png'),attacked_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',  opt.output_dir,name_to_test, 'annot_result/original_'+name+'_'+str(preprocess_num)+'.png'),original_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
  
                    if rect.dim() == 3:
                        rect_list[name].append(rect[i,0,:].detach().cpu().numpy().tolist())
                    elif rect.dim() == 2:
                        rect_list[name].append(rect[i,:].detach().cpu().numpy().tolist())
                    rect_list['mask_size'].append(size_of_mask)
                    
                    if 'random_mask' in opt.algorithm:
                        assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name_to_test,'random_mask/','random_mask_'+str(preprocess_num)+'.png'),random_mask[i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/inference_mask.png'),mask[i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                preprocess_num+=1
            if preprocess_num>=1000:
                break

    with open('inpainting/ensemble/experiment_result/'+opt.output_dir+'/rect.json','w') as f:
        json.dump(rect_list,f,indent=4)
   
if __name__=='__main__':
    main()
