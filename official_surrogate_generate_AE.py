# from inpainting.RFR_Inpainting.run import attack
from inpainting.ensemble.official_Attack import Attacker
import argparse
from inpainting.ensemble.source_models.FcF import FcFnetAPI
import torch
from inpainting.ensemble.official_load_models import load_models
import cv2

import kornia as K
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader

from inpainting.ensemble.official_dataset import AttackDataset,TransferAttackDataset
from inpainting.ensemble.options import get_args
from pathlib import Path

import json
import os

import kornia as K
import time

# run
# python -m  inpainting.ensemble.official_surrogate_generate_AE --dataset places2 --target_models Matnet --lossType perceptual_loss --InputImg_dir /home1/mingzhi/inpainting/Watermarking --output_dir surrogate_Mat_RPN_refined_mask_baseline_perceptual_loss_watermark_rm_HTX_ensemble_iter=30_e=0.05_mask_fix=0.33_sign --algorithm norm_combine --iter_num 10 --get_logo --attach_logo --budget 0.03 --sign --RPNRefineMask /home1/mingzhi/inpainting/ensemble/HTX_experiment/pred_masks_from_latest_fasterrcnn_mask_instance_mixture --get_logo --attach_logo --validation_model FcFnet --algorithm attribution_attack_for_multi_task3 --algorithm random_logo_alpha

def get_mask_edge(mask):
    edges=K.filters.canny(mask,sigma=(2,2),low_threshold=0.05,high_threshold=0.1,kernel_size=(21,21))[1]
    return edges

def main():
    
    opt=get_args()

    # RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet=load_models(opt)
    RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet, WDModel, DBWEModel, SLBRModel=load_models(opt)
    # del RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet
    # del WDModel, DBWEModel, SLBRModel
    # del RFRnet, GMCNNnet, EdgeConnet, DBWEModel
    del RFRnet, GMCNNnet, EdgeConnet, 

    # all_models={'RFRnet':RFRnet,'GMCNNnet':GMCNNnet,'EdgeConnet':EdgeConnet,'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet}
    # all_models={'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet}
    # all_models = { 'WDModel':WDModel, "DBWEModel":DBWEModel, 'SLBRModel':SLBRModel}
    # all_models = { 'WDModel':WDModel,'SLBRModel':SLBRModel, "DBWEModel":DBWEModel,}
    # all_models = { 'WDModel':WDModel,'SLBRModel':SLBRModel,}
    # all_models = {'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet,  'SLBRModel':SLBRModel, "DBWEModel":DBWEModel, }
    all_models = {'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet,   "WDModel":WDModel}
    # all_models = {'FcFnet':FcFnet,'Matnet':Matnet,  'SLBRModel':SLBRModel, "WDModel":WDModel,  }
    # all_models = {'FcFnet':FcFnet, 'SLBRModel':SLBRModel}
    # all_models = {'FcFnet':FcFnet,'Matnet':Matnet,  'SLBRModel':SLBRModel, "DBWEModel":DBWEModel,  }
    # all_models = {'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet,  'SLBRModel':SLBRModel, "DBWEModel":DBWEModel, 'GMCNNnet':GMCNNnet,}
    # all_models = {'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet,  'SLBRModel':SLBRModel,  }
    # all_models = {'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet,  'WDModel':WDModel, "DBWEModel":DBWEModel, }
    # all_models = {'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet,  'WDModel':WDModel, "SLBRModel":SLBRModel, }

    # all_models={'Crfillnet':Crfillnet,'Matnet':Matnet}
    # all_models={'FcFnet':FcFnet,'Matnet':Matnet}

    # RFRnet,GMCNNnet,Crfillnet,EdgeConnet=load_models()
    
    # load models and validation models
    for target_model in opt.target_models:
        models={}
        validation_model={}
        for name in all_models:
            if name!=target_model and name not in opt.validation_model:
                models[name]=all_models[name]
            if name in opt.validation_model:
                validation_model[name]=all_models[name]

        # create experimentd directory
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'annot_result/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'attacked_input/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'attacked_mask/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'original_input/')).mkdir(parents=True,exist_ok=True)
        
        print('after loading model')
        dataset=AttackDataset(opt,opt.InputImg_dir,relative_mask_size=0.33,if_sort=True)

        dataloader=DataLoader(dataset,batch_size=1,shuffle=False)

        attacker=Attacker(opt,models,validation_model=validation_model,iters=opt.iter_num, budget=opt.budget)

        rect_list={}
        # all_information={'rect_list'}
        for name in models.keys():
            rect_list[name]=[]
        preprocess_num=0

        print('start attack', len(dataloader))
        
        # produce adversarial examples for each images in the dataloader
        for index,(img, mask,rect,target_patch, logo) in enumerate(iter(dataloader)):
        # for index,(img, mask,rect,target_patch) in enumerate(iter(dataloader)):
            # if index<757: # wrong surrogate model for our method on Matnet
            #     preprocess_num+=1
            #     continue
            # if index<300:
            #     preprocess_num+=1
            #     continue
            # if index>=200:
            #     break
            # if index<473:
            #     preprocess_num+=1
            #     continue
            # if index<444:
            #     preprocess_num+=1
            #     continue
            # if index == 200:
            #     break
            # if index <= 15:
            #     preprocess_num+=1
            #     continue
            

            img=img.cuda()
            mask=mask.cuda()
            if opt.target_attack and opt.target_img!='':
                target_patch=target_patch.cuda()

            print('start to generate original output')
            original_output={}
            original_FeatList={}
            original_pred_mask = {}
            attacked_pred_mask = {}
            validation_output=None
            original_output_for_validation=None
            
            with torch.no_grad():
                print('rect', rect)
                for name in models.keys():
                    rect_list[name].append(rect[0,:].detach().cpu().numpy().tolist())

                    # output original result
                    if 'lowlevel_perceptual_loss'  not in   opt.lossType and   'similarity_inverse_loss' not in opt.lossType:
                        if name in [ 'WDModel', 'DBWEModel', 'SLBRModel']:
                            original_output[name]=models[name](img)
                        else:
                            original_output[name]=models[name](img,mask)
                    elif 'lowlevel_perceptual_loss' in  opt.lossType or 'similarity_inverse_loss' in opt.lossType:
                        original_output[name], original_FeatList[name]=models[name](img,mask,keepFeat=[0,1,2,3])
                    
                print('rect list', rect_list)
            
            # continue

            with torch.no_grad():

                # optional target attack
                if opt.target_attack and opt.target_img!='':
                    target_patch=transforms.Resize((rect[0,0,1],rect[0,0,3]))(target_patch)
                elif opt.target_attack and opt.target_img=='' and 'to_bright' in opt.attackType:
                    target_patch=torch.ones((img.size()[0],img.size()[1],rect[0,0,1],rect[0,0,3])).cuda()

                output_mask=None

                # target attack
                if opt.target_attack:
                    if 'unchange_attack' in opt.attack_type:
                        target_patch=img[:,:,rect[0,0,0]:rect[0,0,1]+rect[0,0,0],rect[0,0,2]:rect[0,0,2]+rect[0,0,3]]
                        input_with_final_noise,adv_noise,attacked_output=attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)
                    elif opt.attack_specific_region:
                        # print('start to attack')
                        input_with_final_noise,adv_noise,attacked_output=attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)    
                    else:
                        input_with_final_noise,adv_noise,attacked_output,output_mask=attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)
                
                # untarget attack
                else:
                    if opt.validation_model=={}:
                        input_with_final_noise,adv_noise,attacked_output,output_mask=attacker.attack(img,original_output,mask,rect=rect,original_FeatList=original_FeatList)
                    else:
                        input_with_final_noise,adv_noise,attacked_output,output_mask,validation_output,original_output_for_validation,validation_score=attacker.attack(img,original_output,mask,rect=rect,original_FeatList=original_FeatList, index = index)
                    
                
                # get the original output 0 and mask 1
                original_pred_mask = {key: value[1] if key in [ 'WDModel', 'DBWEModel', 'SLBRModel'] else value for key, value in original_output.items() }
                attacked_pred_mask = {key: value[1] if key in [ 'WDModel', 'DBWEModel', 'SLBRModel'] else value for key, value in attacked_output.items() }
          
                original_output = {key: value[0] if key in [ 'WDModel', 'DBWEModel', 'SLBRModel'] else value for key, value in original_output.items() }
                attacked_output = {key: value[0] if key in [ 'WDModel', 'DBWEModel', 'SLBRModel'] else value for key, value in attacked_output.items() }
                if opt.validation_model != {}:
                    validation_pred_mask={key: value[0] if key in [ 'WDModel', 'DBWEModel', 'SLBRModel'] else value for key, value in validation_output.items()}
                    validation_output={key: value[0] if key in [ 'WDModel', 'DBWEModel', 'SLBRModel'] else value for key, value in validation_output.items()}
                    

              
                
                annot_attacked_output={}


                

                
                for i in range(attacked_output[list(attacked_output.keys())[0]].size()[0]):
                    
                    if output_mask!=None:
                        output_mask = output_mask.cuda()
                        edge=get_mask_edge(output_mask) # get the edge of mask
                        original_output={}
                        original_FeatList={}
                        with torch.no_grad():
                            for name in models.keys():
                                print(name,'to play')

                                # get the original models again with the output masks
                                if 'lowlevel_perceptual_loss'  not in   opt.lossType and   'similarity_inverse_loss' not in opt.lossType:
                                    if name in [ 'WDModel', 'DBWEModel', 'SLBRModel']:
                                        original_output[name]=models[name](img)
                                        original_pred_mask[name] = original_output[name][1]
                                        original_output[name] = original_output[name][0]
                                    else:     
                                        original_output[name]=models[name](img,output_mask)

                                elif 'lowlevel_perceptual_loss' in  opt.lossType or 'similarity_inverse_loss' in opt.lossType:
                                    original_output[name], original_FeatList[name]=models[name](img,output_mask,keepFeat=[0,1,2,3])
       
                    
                    

                    for name in attacked_output.keys():
                        # save adv result, original result and the different btw them
                        assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/attacked_'+name+'_'+str(preprocess_num)+'.png'),attacked_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                        assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/original_'+name+'_'+str(preprocess_num)+'.png'),original_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                        assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/diff_'+name+'_'+str(preprocess_num)+'.png'),torch.abs(original_output[name]-attacked_output[name])[i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])

                        assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/output_with_mask_'+name+'_'+str(preprocess_num)+'.png'),((transforms.Resize(attacked_output[name][i].size()[-2:])(output_mask)[i]>0.9)*attacked_output[name][i]).detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                        
                       
                        # save images with box of mask
                        if output_mask!=None:
                            attacked_output[name]=torch.clamp(attacked_output[name]+transforms.Resize(attacked_output[name].size()[-2:])(edge),min=0,max=1)

                        assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'annot_result/attacked_'+name+'_'+str(preprocess_num)+'.png'),attacked_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                        assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'annot_result/original_'+name+'_'+str(preprocess_num)+'.png'),original_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])

                        # if the model is blind watermark remover, also save the detected mask of watermark
                        if name in [ 'WDModel', 'DBWEModel', 'SLBRModel']:
                            assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'annot_result/original_pred_mask_'+name+'_'+str(preprocess_num)+'.png'),original_pred_mask[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                            assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'annot_result/attacked_pred_mask_'+name+'_'+str(preprocess_num)+'.png'),attacked_pred_mask[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])

                    
                        # rect_list[name].append(rect[i,:].detach().cpu().numpy().tolist())
                    
                    # save mask
                    if output_mask!=None:
                        assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'attacked_mask/mask_'+'_'+str(preprocess_num)+'.png'),(output_mask[i]>0.9).detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                        
                        
                    # save adversarial noise
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'attacked_input/',str(preprocess_num)+'.png'),input_with_final_noise[i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    
                    # save original input
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'original_input/',str(preprocess_num)+'.png'),img[i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    preprocess_num+=1

            
            del original_output, original_FeatList, original_pred_mask, attacked_pred_mask, validation_output
            # exit()
            torch.cuda.empty_cache()
            
            if preprocess_num>=1000:
                break
        
        # save information of mask in rect.
        with open(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'rect.json'),'w') as f:
            json.dump(rect_list,f,indent=4) 
     
if __name__=='__main__':
    main()
