# from inpainting.RFR_Inpainting.run import attack
from inpainting.ensemble.Attack import Attacker,generate_rect_mask
import argparse
from inpainting.ensemble.source_models.FcF import FcFnetAPI
import torch
from inpainting.ensemble.load_models import load_models
import cv2

import kornia as K
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader

from inpainting.ensemble.dataset import AttackDataset,TransferAttackDataset
from inpainting.ensemble.options import get_args
from pathlib import Path

import json
import os

import kornia as K

import time 
import random
from inpainting.ensemble.utils import rbf_interpolation, ND_interp

from scipy.optimize import differential_evolution, rosen

def get_mask_edge(mask):
    # images_grey=transforms.functional.rgb_to_grayscale(mask)
    edges=K.filters.canny(mask,sigma=(2,2),low_threshold=0.05,high_threshold=0.1,kernel_size=(21,21))[1]
    return edges


# from inpainting.ensemble.utils import rbf_interpolation
from scipy.stats import rv_discrete


from inpainting.ensemble.utils import rbf_interpolation, ND_interp,get_superpixel,get_semantic_loss

# evaluation func for evolution
def evaluation_for_evolution(input_parameter, img, rect, original_output, DE_records, best_validation_score, search_size, relative_mask_size, opt, original_FeatList, mask_size , mask_score_map, activated_mask, target_patch, attacker):
    mask_index=[0,0]
    

    mask_index[0]=int(input_parameter[0]*(search_size[0]-1))
    mask_index[1]=int(input_parameter[1]*(search_size[1]-1))
    
    DE_records['box_list'].append([*(mask_index),relative_mask_size,relative_mask_size])

    mask,tmp_rect=generate_rect_mask(img.size()[-2:],mask_size= (int(mask_size[0]*relative_mask_size), int(mask_size[1]*relative_mask_size)),position=mask_index,rand_mask=False,batch_size=img.size()[0])
    mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()
    if opt.target_attack:

        # try to let the output to remain the same as the origianl one
        if 'unchange_attack' in opt.attack_type:
            target_patch=img[:,:,rect[0,0,0]:rect[0,0,1]+rect[0,0,0],rect[0,0,2]:rect[0,0,2]+rect[0,0,3]]
            input_with_final_noise,adv_noise,attacked_output=attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)
        # attack specific region
        elif opt.attack_specific_region:
            input_with_final_noise,adv_noise,attacked_output=attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)    
        else:
            input_with_final_noise,adv_noise,attacked_output,output_mask=attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)
    else:    
        # untarget attack
        tmp_attacked_output,attacked_score=attacker.evaluate(img,None,original_output,mask,rect=rect,original_FeatList=original_FeatList)

    # tmp_attacked_output=None
    # use superpixel map as the semantic proxy
    if 'semantic_attack' in opt.algorithm:
        semantic_loss=get_semantic_loss(superpixel_map,mask)
        attacked_score=semantic_loss #overwrite the output of attack
    if isinstance(attacked_score, torch.Tensor):
        attacked_score = attacked_score.item()

    # record the intermediate output of DE, save them in the list
    DE_records['score_list'].append(attacked_score)
    # mark the score on the map with the location of the mask
    mask_score_map[mask_index[0],mask_index[1]]=attacked_score
    # mark the activation on the map with the location of the mask
    activated_mask[mask_index[0],mask_index[1]]=1
    # print('validation score',attacked_score,best_validation_score)
    # print('mask score map',torch.sum(mask_score_map))

    # save the best mask
    if attacked_score<=best_validation_score:
        best_validation_score=attacked_score
        attacked_output=tmp_attacked_output
        output_mask=mask

    return attacked_score
def main():
    
    opt=get_args()

    RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet=load_models(opt)

    # all_models={'RFRnet':RFRnet,'GMCNNnet':GMCNNnet,'EdgeConnet':EdgeConnet,'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet}
    all_models={'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet, 'RFRnet':RFRnet}

    
    for target_model in opt.target_models:
        # if target_model=='GMCNNnet' or target_model=='RFRnet':
        #     continue
        models={}
        validation_model={}
        for name in all_models:
            if name!=target_model and name not in opt.validation_model:
                models[name]=all_models[name]
            if name in opt.validation_model:
                validation_model[name]=all_models[name]
    
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'annot_result/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'attacked_input/')).mkdir(parents=True,exist_ok=True)
        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'attacked_mask/')).mkdir(parents=True,exist_ok=True)

        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'validation_output/')).mkdir(parents=True,exist_ok=True)

        Path(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'DE_records/')).mkdir(parents=True,exist_ok=True)

       
        print('after loading model')

        # load dataset
        relative_mask_size=0.33
        if opt.dataset=='places2' and False:
            dataset=AttackDataset(opt,opt.InputImg_dir,relative_mask_size=relative_mask_size,if_sort=False)
        else:
            dataset=AttackDataset(opt,opt.InputImg_dir,relative_mask_size=relative_mask_size,if_sort=True)
            
       
        dataloader=DataLoader(dataset,batch_size=1,shuffle=False)

        attacker=Attacker(opt,models,validation_model=validation_model,iters=opt.iter_num, budget=opt.budget)

        rect_list={}
        # all_information={'rect_list'}
        for name in models.keys():
            rect_list[name]=[]
        preprocess_num=0

        print('start attack')
        
        average_time_cost=0
        for index,(img, mask,rect,target_patch) in enumerate(iter(dataloader)):
        # for index,(img, mask,rect,target_patch) in enumerate(iter(dataloader)):
    
            if index<4000:
                preprocess_num+=1
                continue
        
            filename=dataset.filename_list[index]

            img=img.cuda()
            mask=mask.cuda()
            if opt.target_attack and opt.target_img!='':
                target_patch=target_patch.cuda()
         

            print('start to generate original output')
            original_output={}
            original_FeatList={}
            
            # necessary to get the superpixel map first
            if 'semantic_attack' in opt.algorithm:
                superpixel_map,superpixel_map_vis=get_superpixel(img)
            
            with torch.no_grad():
                for name in models.keys():
                    print(name,'to play')
     
                    # generate original output
                    if 'lowlevel_perceptual_loss'  not in   opt.lossType and   'similarity_inverse_loss' not in opt.lossType:
                        original_output[name]=transforms.Resize(models[name](img,mask).size()[-2:])(img)
                    elif 'lowlevel_perceptual_loss' in  opt.lossType or 'similarity_inverse_loss' in opt.lossType:
                        original_output[name], original_FeatList[name]=models[name](img,mask,keepFeat=[0,1,2,3])
                    
            
            if opt.target_attack and opt.target_img!='':
                target_patch=transforms.Resize((rect[0,0,1],rect[0,0,3]))(target_patch)
            elif opt.target_attack and opt.target_img=='' and 'to_bright' in opt.attackType:
                target_patch=torch.ones((img.size()[0],img.size()[1],rect[0,0,1],rect[0,0,3])).cuda()

            # initialize necessary index records
            mask_size=mask.size()[-2:]
            search_size=(int(img.size()[-2]-mask.size()[-2]*relative_mask_size),int(img.size()[-1]-mask.size()[-1]*relative_mask_size))
            mask_score_map=torch.zeros(search_size).cuda()
            activated_mask=torch.zeros(search_size).cuda()
            num_of_attack=0

            # initialize DE records
            DE_records={'id_list':filename,'score_list':[],'box_list':[]}
        

            if 'random_search' in opt.algorithm or True:
                mask_index=[0,0]
                global best_validation_score
                best_validation_score=0
      
                output_mask=None

                T1=time.time()

                # initial necessary args for DE algorithm
                bound_for_input_parameter=[(0,1),(0,1)]
                args_for_DE_obj = (img, rect, original_output, DE_records, best_validation_score, search_size, relative_mask_size, opt, {}, mask_size, mask_score_map, activated_mask, None, attacker)

                # do evolution algorithm
                result=differential_evolution(func=evaluation_for_evolution,
                                    bounds=bound_for_input_parameter, 
                                    args=args_for_DE_obj,
                          
                                    )
                print('len args', len(args_for_DE_obj))

                # get the size of final DE result
                mask_index[0]=int(result.x[0]*(search_size[0]-1))
                mask_index[1]=int(result.x[1]*(search_size[1]-1))
                
                # generate mask from the DE result
                mask,tmp_rect=generate_rect_mask(img.size()[-2:],mask_size= (int(mask_size[0]*relative_mask_size), int(mask_size[1]*relative_mask_size)),position=mask_index,rand_mask=False,batch_size=img.size()[0])
                output_mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()

                # evaluate the optimal mask from DE
                attacked_output,attacked_score=attacker.evaluate(img,None,original_output,output_mask,rect=rect,original_FeatList=original_FeatList)
                
                print('validation scre',attacked_score,'best_validation_score',best_validation_score,attacked_score<=best_validation_score)
                T2=time.time()
                time_cost=(T2-T1)
                print('time cost', time_cost)
                average_time_cost+=time_cost


                num_of_attack+=1
                
                # get the score map of locations of maps uniformly by interpolation 
                intepolated_maks_score_map,inter_result_to_show=ND_interp(mask_score_map,activated_mask)
                    
            
            
            annot_attacked_output={}


            

            
            for i in range(attacked_output[list(attacked_output.keys())[0]].size()[0]):
                
                # if get optimal mask, then regenerate the outputs of removers with the masks again.
                if output_mask!=None:
                    edge=get_mask_edge(output_mask)
                    original_output={}
                    original_FeatList={}
                    with torch.no_grad():
                        for name in models.keys():

                            if 'lowlevel_perceptual_loss'  not in   opt.lossType and   'similarity_inverse_loss' not in opt.lossType:
                                original_output[name]=models[name](img,output_mask)
                            elif 'lowlevel_perceptual_loss' in  opt.lossType or 'similarity_inverse_loss' in opt.lossType:
                                original_output[name], original_FeatList[name]=models[name](img,output_mask,keepFeat=[0,1,2,3])
 
                # save the original remover outputs, adv outputs and their difference.
                for name in attacked_output.keys():
                    
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/attacked_'+name+'_'+str(preprocess_num)+'.png'),attacked_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/diff_'+name+'_'+str(preprocess_num)+'.png'),torch.abs(original_output[name]-attacked_output[name])[i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/output_with_mask_'+name+'_'+str(preprocess_num)+'.png'),((transforms.Resize((attacked_output[name  ][i].size()[-2:]))(output_mask)[i]>0.9)*attacked_output[name][i]).detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    
                   
                    # draw the box of mask and save the images
                    if output_mask!=None:
                        attacked_output[name]=torch.clamp(attacked_output[name]+transforms.Resize((attacked_output[name].size()[-2:]))(edge),min=0,max=1)

                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'annot_result/attacked_'+name+'_'+str(preprocess_num)+'.png'),attacked_output[name][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
     

                    

                    
    
                
                    rect_list[name].append(rect[i,0,:].detach().cpu().numpy().tolist())
                
                assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/original_'+'_'+str(preprocess_num)+'.png'),original_output[list(original_output.keys())[0]][i].detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])

                if output_mask!=None:
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'attacked_mask/mask_'+'_'+str(preprocess_num)+'.png'),(output_mask[i]>0.9).detach().cpu().numpy().transpose(1,2,0)*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    
                    
                # save the score map
                if 'semantic_attack' in opt.algorithm:
                    if output_mask!={}:
                        superpixel_map_vis=np.clip(superpixel_map_vis+transforms.Resize((superpixel_map_vis.shape[-2:]))(edge)[0].detach().cpu().numpy(),a_min=0,a_max=1)

                    
                    assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'annot_result/superpixel_map_'+str(preprocess_num)+'.png'),superpixel_map_vis.transpose(1,2,0)*255)
                
                assert cv2.imwrite(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'result/mask_score_map_'+name+'_'+str(preprocess_num)+'.png'),inter_result_to_show*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                preprocess_num+=1
                print('preprocess num',preprocess_num)
            
            # save DE records including the intermediate results
            with open(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'DE_records', filename+'.json'),'w') as f:
                json.dump(DE_records,f,indent=4)
                print('filename',filename)
                print('DE_records',os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'DE_records', filename+'.json'))
         
        
        print('average_time_cost',average_time_cost/num_of_attack)
        rect_list['average_time_cost']=average_time_cost/num_of_attack
        with open(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,target_model,'rect.json'),'w') as f:
            json.dump(rect_list,f,indent=4)
          
if __name__=='__main__':
    main()
