# from advDF.ensemble_test import attacks
# from inpainting.Perceptual_loss_for_real_time_style_transfer.train import calc_Content_Loss, calc_Gram_Loss
# from inpainting.edge_connect.src.loss import PerceptualLoss
# from inpainting.RFR_Inpainting.run import attack
from dis import dis
from re import X
from builtins import isinstance
# from inpainting.RFR_Inpainting.run import attack
import torch
# from inpainting.inpainting_gmcnn.pytorch.noise_fn import Noise_Func
from inpainting.ensemble.official_noise_func import Noise_Func
from torch.optim.lr_scheduler import StepLR
from inpainting.inpainting_gmcnn.pytorch.util.utils import generate_rect_mask, generate_stroke_mask, getLatest
import cv2
import torch.optim as optim
import torchvision
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np

from pathlib import Path

from inpainting.ensemble.clustering_measure import ADMM_attacker
from tqdm import tqdm as tqdm
import warnings

import os
warnings.filterwarnings("ignore")
figure_for_paper_index = 0

# attacker class
class Attacker():  
    def __init__(self,opt,target_models,validation_model={},lr=0.02,iters=150,budget=0.05):
        self.opt=opt
        self.target_models=target_models
        self.validation_models=validation_model
        if self.opt.target_attack:
            self.lr=budget/50
        else:
            self.lr=lr
       
        self.iters=iters
        self.budget=budget if budget != None else 0.05

        # VGG model to calculate perceptual loss during attack
        self.loss_vgg=torchvision.models.__dict__['vgg16'](pretrained=True).features.to('cuda')
        self.vgg_style_layers=[3,8,15,22]
        self.vgg_content_layers=[15]

        self.mse_criterion=torch.nn.MSELoss(reduction='mean')
        self.inpainting_model_names = ['RFRnet','GMCNNnet','EdgeConnet','Crfillnet','Gennet','FcFnet','Matnet']
        self.watermark_remover_names = ['WDModel', "DBWEModel", 'SLBRModel']
        

        self.device='cuda'
        
    # pixel-wise loss D(output1, output2)
    def output_pixel_loss(self,original_output,attacked_output,mask=None):
        mse_fn=torch.nn.MSELoss()
        loss=[]

        for name in attacked_output.keys():
            
            # just care about the pixels inside the mask, which is the common way
            if self.opt.attack_specific_region or mask is not None:
               
                tmp_loss=-mse_fn(original_output[name]*( torchvision.transforms.Resize((original_output[name].size()[-2],original_output[name].size()[-1]))(mask)),attacked_output[name]*( torchvision.transforms.Resize((original_output[name].size()[-2],original_output[name].size()[-1]))(mask)))

                # for debugging nan values
                if torch.sum(torch.isnan(tmp_loss))>0:
                    print('original_output[name]',torch.sum(torch.isnan(original_output[name])),'mask',torch.sum(torch.isnan(mask)),'attacked_output[name]',torch.sum(torch.isnan(attacked_output[name])))
                    exit()
                    
                loss.append(tmp_loss)
            
            # care about the differences on the entire image 
            else:
                if isinstance(original_output, dict):
                    loss.append(-mse_fn(original_output[name],attacked_output[name]))
                else:
                    loss.append(-mse_fn(original_output,attacked_output[name]))
        if 'MaxLB' in self.opt.algorithm:
            # print(loss)
            loss=max(loss)
        else:
            # print(loss)
            loss=sum(loss)
        print('before reshape',loss)
        return loss

    # cal the content loss in perceptual loss
    def calc_Content_Loss(self,features, targets, weights=None):
        if weights is None:
            weights = [1/len(features)] * len(features)
        # print(weights)
        content_loss = 0
        for name in features.keys():
            tmp_content_loss=0
            for f, t, w in zip(features[name], targets[name], weights):

                tmp_content_loss += self.mse_criterion(f, t) * w
            # print(name,tmp_content_loss)
            content_loss+=tmp_content_loss
            
        return content_loss

    def gram(self,x):
        b ,c, h, w = x.size()
        g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1,2))
        return g.div(h*w)
    def calc_Gram_Loss(self,features, targets, weights=None):
        
        if weights is None:
            weights = [1/len(features)] * len(features)
            
        gram_loss = 0
        for name in features.keys():
            for f, t, w in zip(features[name], targets[name], weights):
                gram_loss += self.mse_criterion(self.gram(f), self.gram(t)) * w
        return gram_loss


    # extract features from each layers of the model
    def extract_features(self, model, x, layers,mask=None,rect=None):
        All_features={}
        
        for name in x.keys():
            features = list()
            # if self.opt.attack_specific_region:
            if rect != None:
                # single_x=x[name]*(1-mask)
                single_x=x[name][:,:,rect[0,0]:rect[0,0]+rect[0,1],rect[0,2]:rect[0,2]+rect[0,3]]
                assert cv2.imwrite('inpainting/ensemble/intermediate_output/extracted_region.jpg',single_x[0].detach().cpu().numpy().transpose(1,2,0)*255)

            else:
                single_x=x[name]
            if not isinstance(model,dict):
                for index, layer in enumerate(model):
                    single_x= layer(single_x)
                    if index in layers:
                        features.append(single_x)
            else:
                for index, layer in enumerate(model[name]):
                    single_x= layer(single_x)
                    if index in layers:
                        features.append(single_x) 
            All_features[name]=features
        return All_features
    

    """ def get_gradient_loss(self,original_output,attacked_output,mask=None):
        mse_fn=torch.nn.MSELoss()
        loss=[]
        for name in self.target_models.keys():

            if self.opt.attack_specific_region:

                loss.append(-mse_fn(original_output[name]*(1-mask),attacked_output[name]*(1-mask)))
            else:
                loss.append(-mse_fn(original_output[name],attacked_output[name]))
        if 'MaxLB' in self.opt.algorithm:
            print(loss)
            loss=max(loss)
        else:
            print(loss)
            loss=sum(loss)
        print('before reshape',loss)
        return loss """

    # calculate the perceptual loss using the components defined above
    def get_perceputal_loss(self,original_output,tmp_output,noise_func=None,rect=None):
        original_output = {key:value if not isinstance(value, tuple) else value[0] for key, value in original_output.items()  }
        tmp_output = {key:value if not isinstance(value, tuple) else value[0] for key, value in tmp_output.items()  }
        
        perceptual_loss=0
        

        original_output_content_features=self.extract_features(self.loss_vgg,original_output,self.vgg_content_layers,rect=rect)
        attacked_output_content_features=self.extract_features(self.loss_vgg,tmp_output,self.vgg_content_layers,rect=rect)
        
        original_output_style_features=self.extract_features(self.loss_vgg,original_output,self.vgg_style_layers,rect=rect)
        attacked_output_style_features=self.extract_features(self.loss_vgg,tmp_output,self.vgg_style_layers,rect=rect)
        
        
        perceptual_loss+=1.0*self.calc_Content_Loss(original_output_content_features,attacked_output_content_features)
        perceptual_loss+=3.0*self.calc_Gram_Loss(original_output_style_features,attacked_output_style_features)
        
        if 'mse_noise' in self.opt.lossType and self.opt.target_attack:
            perceptual_loss+=4*self.mse_criterion(noise_func.delta_x,torch.zeros(noise_func.delta_x.size()).cuda())
        if 'mse_noise' in self.opt.lossType and not self.opt.target_attack:
            perceptual_loss-=4*self.mse_criterion(noise_func.delta_x,torch.zeros(noise_func.delta_x.size()).cuda())

        perceptual_loss = -perceptual_loss
        return perceptual_loss


    # get the loss on the low level features of inpainting models themselves
    def get_lowlevel_perceputal_loss_inpainting_model(self,original_FeatList,attacked_FeatList,noise_func=None,rect=None):
        perceptual_loss=0

        
        if 'CosineLoss' in self.opt.lossType:
            # if 'MaxLB' in self.opt.algorithm:
            #     cosineLoss=[]
            # else:
            #     cosineLoss=0
            cosineLoss=[]
            for name in original_FeatList.keys():
                
                if False:
                    if isinstance(attacked_output_content_features,dict):
                        for index in range(len(original_output_content_features[name])):
                            # if 'MaxLB' in self.opt.algorithm:
                            cosineLoss.append(torch.sum(original_output_content_features[name][index]*attacked_output_content_features[name][index])+3*torch.sum(original_output_style_features[name][index]*attacked_output_style_features[name][index]))
                            # else:
                            #     cosineLoss+=torch.sum(original_output_content_features[name][index]*attacked_output_content_features[name][index])
                            #     cosineLoss+=3*torch.sum(original_output_style_features[name][index]*attacked_output_style_features[name][index])
                    else:
                        for index in range(len(original_output_content_features[name])):
                            cosineLoss.append(torch.sum(original_output_content_features[name][index]*attacked_output_content_features[index] )+3*torch.sum(original_output_style_features[name][index]*attacked_output_style_features[index] ))
                            # cosineLoss+=3*torch.sum(original_output_style_features[name][index]*attacked_output_style_features[index] )
                if True:
                    cos_fn=torch.nn.CosineSimilarity(dim=1)
                    tmp_cosineLoss=0
                    if isinstance(attacked_FeatList,dict):
                        for index in range(len(original_FeatList[name])):

                            flatten_original_output_features=torch.flatten( original_FeatList[name][index] ,start_dim=1)
                            flatten_attacked_output_features=torch.flatten(attacked_FeatList[name][index] ,start_dim=1)
            
                            tmp_cosineLoss+=(torch.sum(cos_fn(flatten_original_output_features, flatten_attacked_output_features )) )

                    else:
                        for index in range(len(original_FeatList[name])):
                            flatten_original_output_features=torch.flatten( original_FeatList[name][index] ,start_dim=1)
                            flatten_attacked_output_features=torch.flatten(attacked_FeatList[name]  ,start_dim=1)
 
                            tmp_cosineLoss+=(torch.sum(cos_fn(flatten_original_output_features, flatten_attacked_output_features )) )
            
                
                cosineLoss.append(tmp_cosineLoss)
            if 'MaxLB' in self.opt.algorithm:
                print('list',cosineLoss)
                cosineLoss=max(cosineLoss)
            else:
                cosineLoss= sum(cosineLoss)
            perceptual_loss-=cosineLoss
        else:
            perceptual_loss+=1.0*self.calc_Content_Loss(original_FeatList,attacked_FeatList)
    
        
        if 'mse_noise' in self.opt.lossType and self.opt.target_attack:
            perceptual_loss+=4*self.mse_criterion(noise_func.delta_x,torch.zeros(noise_func.delta_x.size()).cuda())
        if 'mse_noise' in self.opt.lossType and not self.opt.target_attack:
            perceptual_loss-=4*self.mse_criterion(noise_func.delta_x,torch.zeros(noise_func.delta_x.size()).cuda())
        return perceptual_loss

    # get the perceptual loss of lowlevel features 
    def get_lowlevel_perceputal_loss(self,original_output,tmp_output,noise_func=None,rect=None):
        perceptual_loss=0
        
        original_output_content_features=self.extract_features(self.loss_vgg,original_output,[0,1,2,3],rect=rect)
        attacked_output_content_features=self.extract_features(self.loss_vgg,tmp_output,[0,1,2,3],rect=rect)
        
        original_output_style_features=self.extract_features(self.loss_vgg,original_output,[0,1,2,3],rect=rect)
        attacked_output_style_features=self.extract_features(self.loss_vgg,tmp_output,[0,1,2,3],rect=rect)
        
        if 'CosineLoss' in self.opt.lossType:

            cosineLoss=[]
            for name in original_output.keys():
                
                if False:
                    if isinstance(attacked_output_content_features,dict):
                        for index in range(len(original_output_content_features[name])):
                            # if 'MaxLB' in self.opt.algorithm:
                            cosineLoss.append(torch.sum(original_output_content_features[name][index]*attacked_output_content_features[name][index])+3*torch.sum(original_output_style_features[name][index]*attacked_output_style_features[name][index]))
                            # else:
                            #     cosineLoss+=torch.sum(original_output_content_features[name][index]*attacked_output_content_features[name][index])
                            #     cosineLoss+=3*torch.sum(original_output_style_features[name][index]*attacked_output_style_features[name][index])
                    else:
                        for index in range(len(original_output_content_features[name])):
                            cosineLoss.append(torch.sum(original_output_content_features[name][index]*attacked_output_content_features[index] )+3*torch.sum(original_output_style_features[name][index]*attacked_output_style_features[index] ))
                            # cosineLoss+=3*torch.sum(original_output_style_features[name][index]*attacked_output_style_features[index] )
                if True:
                    cos_fn=torch.nn.CosineSimilarity(dim=1)
                    tmp_cosineLoss=0
                    if isinstance(attacked_output_content_features,dict):
                        for index in range(len(original_output_content_features[name])):
                            # if 'MaxLB' in self.opt.algorithm:
                            flatten_original_output_features=torch.flatten( original_output_content_features[name][index] ,start_dim=1)
                            flatten_attacked_output_features=torch.flatten(attacked_output_content_features[name][index] ,start_dim=1)
                            flatten_attacked_output_style_features=torch.flatten(attacked_output_style_features[name][index] ,start_dim=1)
                            flatten_origianl_output_style_features=torch.flatten(original_output_style_features[name][index] ,start_dim=1)
                            tmp_cosineLoss+=(torch.sum(cos_fn(flatten_original_output_features, flatten_attacked_output_features ))+3*torch.sum(cos_fn( flatten_attacked_output_style_features ,flatten_origianl_output_style_features)  ))
   
                    else:
                        for index in range(len(original_output_content_features[name])):
                            flatten_original_output_features=torch.flatten( original_output_content_features[name][index] ,start_dim=1)
                            flatten_attacked_output_features=torch.flatten(attacked_output_content_features[name]  ,start_dim=1)
                            flatten_attacked_output_style_features=torch.flatten(attacked_output_style_features[name]  ,start_dim=1)
                            flatten_origianl_output_style_features=torch.flatten(original_output_style_features[name][index] ,start_dim=1)
                            tmp_cosineLoss+=(torch.sum(cos_fn(flatten_original_output_features, flatten_attacked_output_features ))+3*torch.sum(cos_fn( flatten_attacked_output_style_features ,flatten_origianl_output_style_features)  ))
          
                
                cosineLoss.append(tmp_cosineLoss)
            if 'MaxLB' in self.opt.algorithm:
                print('list',cosineLoss)
                cosineLoss=max(cosineLoss)
            else:
                cosineLoss=torch.sum(cosineLoss)
            perceptual_loss-=cosineLoss
        else:
            perceptual_loss+=1.0*self.calc_Content_Loss(original_output_content_features,attacked_output_content_features)
            perceptual_loss+=3.0*self.calc_Gram_Loss(original_output_style_features,attacked_output_style_features)
        
        if 'mse_noise' in self.opt.lossType and self.opt.target_attack:
            perceptual_loss+=4*self.mse_criterion(noise_func.delta_x,torch.zeros(noise_func.delta_x.size()).cuda())
        if 'mse_noise' in self.opt.lossType and not self.opt.target_attack:
            perceptual_loss-=4*self.mse_criterion(noise_func.delta_x,torch.zeros(noise_func.delta_x.size()).cuda())
        return perceptual_loss
    
    def PatchSim(self,FeatList):
        PATCH_NUM=4
        cossim_fn=torch.nn.CosineSimilarity(dim=1)

        Similarity_List=[]
        for i, featMap in enumerate(FeatList):
            H,W=int(featMap.size()[-2]/PATCH_NUM), int(featMap.size()[-1]/PATCH_NUM)
            DividedFeatList=[featMap[:,:,H*j:H*(j+1),W*k:W*(k+1)] for j in range(PATCH_NUM) for k in range(PATCH_NUM)]
            tmp_Similarity_List=[]
            for j in range(len(DividedFeatList)):
                tmp_Similarity_List1=[]
                for k in range(len(DividedFeatList)):
                    # if j<k:
                    # print(DividedFeatList[j].size(),DividedFeatList[k].size())
                    tmp_Similarity_List1.append(cossim_fn(torch.flatten(DividedFeatList[j],start_dim=1),torch.flatten(DividedFeatList[k],start_dim=1)))
                tmp_Similarity_List.append(tmp_Similarity_List1)
            Similarity_List.append(tmp_Similarity_List)
        return Similarity_List

    
    def triplet_loss(self,attacked_similarity_list, RankingList,m=0.1):
        RankingLoss=0
        for i in range(len(RankingList)-1):
            RankingLoss+=max(0,m+attacked_similarity_list[RankingList[i+1]]-attacked_similarity_list[RankingList[i]])
        
        # print(RankingLoss)
        # exit()
        return RankingLoss
    def similarity_inverse_loss(self,original_FeatList,attacked_FeatList):
        # print('am i here')
        ranking_loss=0
        
        for name in original_FeatList.keys():
                
            original_similarity_matrix=self.PatchSim(original_FeatList[name])
            attacked_similarity_matrix=self.PatchSim(attacked_FeatList[name])

            assert len(original_similarity_matrix)==len(attacked_similarity_matrix)
            def argsort(seq):
                # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
                #lambda version by Tony Veijalainen
                return [x for x,y in sorted(enumerate(seq), key = lambda a: sum(a))]

            
            for i in range(len(original_similarity_matrix)):
                for j in range(len(original_similarity_matrix[i])):
                    # sorted_original_similarity_matrix=original_similarity_matrix[i][j].sort(reverse=True,key=lambda x: torch.sum(x))
                    # sorted_attacked_similarity_matrix=attacked_similarity_matrix[i][j].sort(reverse=True,key=lambda x: torch.sum(x))
                    # print(original_similarity_matrix[i][j])
                    # exit(0)
                    sorted_original_similarity_ranking=argsort(original_similarity_matrix[i][j])
                    ranking_loss+=self.triplet_loss(attacked_similarity_matrix[i][j],sorted_original_similarity_ranking)
                    

        return ranking_loss
            

    """ def ADMM(self,input,noise_func,mask,target_patch,original_FeatList=None):
        if 'norm_combine' in self.opt.algorithm:
            inter_grad={}
        else:
            inter_grad=None

        loss_func=None
        if 'output_pixel_loss' in self.opt.lossType:
            loss_func=None
        
        mask=torch.zeros(mask.size()).cuda()
        # mask=1.0*(torch.rand(mask.size())>0.5).cuda()
        # mask[...,:100,:]=1
        for i in tqdm(range(self.opt.max_search_times),position=1):
            cur_lr_e = self.opt.lr_e
            # cur_lr_g = self.opt.lr_g
            init_params = {'cur_step_g': self.opt.lr_g, 'cur_rho1': self.opt.rho1, 'cur_rho2': self.opt.rho2, 'cur_rho3': self.opt.rho3,'cur_rho4': self.opt.rho4}
            # attack_optimizer.zero_grad()

            # input_with_noise=noise_func(input.clone(),inter_grad=inter_grad)
            
            for mm in range(1,self.opt.maxIter_mm+1):
                # print('mm',mm)
                # self.ADMM_attacker.update_epsilon(input,mask,noise_func,target_patch,loss_func,cur_lr_e)#self,input,mask,noise_func,target_patch,loss_func,lr_e
                mask,cur_lr_g=self.ADMM_attacker.update_G(input,mask,noise_func,target_patch,loss_func,init_params,original_FeatList=original_FeatList)#input,mask,noise_func,target_patch,loss_func,lr_g,init_params
                assert cv2.imwrite('inpainting/ensemble/experiment_result/'+str(mm)+'.jpg',mask.detach().cpu()[0].numpy().transpose(1,2,0)*255)
            mask=(mask>0.5).float()
            print('max mask',torch.max(mask))
            # print('before epsilon','input',torch.sum(torch.isnan(input)),'mask',torch.sum(torch.isnan(input)))
            # self.ADMM_attacker.update_epsilon(input,mask,noise_func,target_patch,loss_func,cur_lr_e)
        
        return mask """
    
    
    def get_loss(self,original_output,tmp_output,mask,noise_func=None,original_FeatList=None,tmp_FeatList=None,rect=None):
        loss=0
        if 'output_pixel_loss' in self.opt.lossType:
            if 'MaskEOT'  in self.opt.algorithm:
          
                loss+=50000*self.output_pixel_loss(original_output,tmp_output,mask)
            else:
          
                loss+=500000*self.output_pixel_loss(original_output,tmp_output,mask)
        if 'perceptual_loss' in self.opt.lossType or 'EG' in self.opt.lossType or 'TIG' in self.opt.lossType or 'TIG_sep' in self.opt.lossType:
            perceptual_loss=self.get_perceputal_loss(original_output,tmp_output,noise_func)
                
            loss+=-perceptual_loss
        
        if 'lowlevel_perceptual_loss' in self.opt.lossType:

            perceptual_loss=1000*self.get_lowlevel_perceputal_loss_inpainting_model(original_FeatList,tmp_FeatList,noise_func)
                
            loss+=-perceptual_loss*1000
        if 'similarity_inverse_loss' in self.opt.lossType:
            loss+=10*self.similarity_inverse_loss(original_FeatList,tmp_FeatList)
        if 'gradient_loss' in self.opt.lossType:
            gradient_loss=self.img_gradient_loss(original_output,tmp_output,rect,mask)
            loss+=-gradient_loss*1000
        if 'optimal_transformation_search' in self.opt.lossType:
            pass
        return loss
    
    # main function for attack
    def attack(self,input,original_output,original_mask,target_patch=None,rect=None,original_FeatList=None, index = None):
        
        # input is the input images dim() = 4
        # original output is the output images of watermark removers dim() = 4
        # original mask is the mask to embed watermark
        # target_patch is the target output for the recovered content of the images, optional for target attack
        # rect is the coordination of the mask
        # original FeatureList is the feature list of the models given the input images
    
        assert torch.is_tensor(input) and torch.is_tensor(original_mask)
        assert input.dim()==4 and original_mask.dim()==4
        input=input.cuda().detach()+torch.rand(input.size()).cuda()*0.02


        ############## get ready for different algorithms may used in attack
        if 'recover_attack' in self.opt.algorithm:
            blank_input=torch.zeros(input.size()).cuda()
            
        original_mask=original_mask.cuda().detach()
        original_input=input.clone().detach()
        

        if self.opt.target_attack:
            assert torch.is_tensor(target_patch) and target_patch.dim()==4

        if 'norm_combine' in self.opt.algorithm:
            inter_grad={}
        else:
            inter_grad=None
        

        # get the adversarial noise func/wrapper for attack
        noise_func=Noise_Func(input.size(),None,None,target_models=self.target_models).cuda()

        # set the optimizer for adversarial noise 
        parameter_to_learn=[{'params':noise_func.delta_x,'lr':self.lr}]
       
        attack_optimizer=optim.SGD(parameter_to_learn)
        scheduler=StepLR(attack_optimizer,step_size=50,gamma=0.5)

        mask= original_mask.clone().detach()

        
        # if 'optimal_mask_search' in self.opt.algorithm:
        #     self.ADMM_attacker=ADMM_attacker(self.target_models,self.opt,self.budget)
        
        # the 1st option for attack for multi task with attribute guidance (not the one proposed in the paper): use the attribute of the benign inputs
        if 'attribution_attack_for_multi_task' in self.opt.algorithm:

            #  calculate the attribute (integrated gradient) for different types of watermark remover
            IG_for_inpainting, tmp_loss = self.integrated_gradient({'original_input':input.clone().detach()},mask,original_output,steps=30, mode = 'inpainting')
            IG_for_watermark_remover, tmp_loss = self.integrated_gradient({'original_input':input.clone().detach()},mask,original_output,steps=30, mode = 'watermark_removal')

            IG_for_inpainting = torch.from_numpy(IG_for_inpainting).cuda()
            

            IG_for_watermark_remover = torch.from_numpy(IG_for_watermark_remover).cuda()
            
            # normalize the attributes from different watermark removers
            IG_for_inpainting = IG_for_inpainting/torch.sum(torch.abs(IG_for_inpainting))
            IG_for_watermark_remover = IG_for_watermark_remover/torch.sum(torch.abs(IG_for_watermark_remover))
            
            # combine the attributes from different types of watermark removers respectively
            IG_for_inpainting_after_norm = torch.abs(IG_for_inpainting)/(torch.abs(IG_for_inpainting)+torch.abs(IG_for_watermark_remover))
            IG_for_watermark_remover_after_norm = torch.abs(IG_for_watermark_remover)/(torch.abs(IG_for_inpainting)+torch.abs(IG_for_watermark_remover))

            IG_for_watermark_remover = IG_for_watermark_remover_after_norm
            IG_for_inpainting = IG_for_inpainting_after_norm
            # assert cv2.imwrite('inpainting/ensemble/IG_for_inpainting.png', (IG_for_inpainting_after_norm)[0].detach().cpu().numpy().transpose(1,2,0)*255)
            # assert cv2.imwrite('inpainting/ensemble/IG_for_watermark_remover.png', (IG_for_watermark_remover_after_norm)[0].detach().cpu().numpy().transpose(1,2,0)*255)
 
        with torch.enable_grad():

            for i in range(self.iters):
                loss=0

                attack_optimizer.zero_grad()
                input.requires_grad=True

                input_with_noise=noise_func(input.clone(),inter_grad=inter_grad)

                for name, fs_model in self.target_models.items():
                    self.target_models[name].zero_grad()
                
                preprocessed_inputs_with_noise=[input_with_noise]

                
                # use Mask EOT algorithm to generate mask-invariant noise (a baseline whose performance is not good)
                if 'MaskEOT'  in self.opt.algorithm:
                    mask, generated_rect = generate_rect_mask((int(input.size()[-2] ),int(input.size()[-1] )), (int(input.size()[-2]*0.5),int(input.size()[-1]*0.5)), rand_mask=True,batch_size=input.size()[0])
                    mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()
                # use input mask
                else:
                    mask= original_mask.clone().detach()
                
                for tmp_input in preprocessed_inputs_with_noise:
                    tmp_output={}
                    tmp_FeatList={}

                    for name,_ in self.target_models.items():

                        # firstly get the current output of the current input (with noise)
                        if 'lowlevel_perceptual_loss' not in self.opt.lossType and   'similarity_inverse_loss' not in self.opt.lossType:
                            
                            if name in [ 'WDModel', 'DBWEModel', 'SLBRModel']:
                                tmp_output[name]=self.target_models[name](tmp_input[name])
            
                            else:
                                tmp_output[name]=self.target_models[name](tmp_input[name],mask)
                                
                            
                        if 'lowlevel_perceptual_loss'   in self.opt.lossType or 'similarity_inverse_loss' in self.opt.lossType:
                            
                            tmp_output[name],tmp_FeatList[name]=self.target_models[name](tmp_input[name],mask,keepFeat=[0,1,2,3])
                    
                    # target attack
                    if self.opt.target_attack:
                        target_output={}
                        for name in tmp_output.keys():
                            
                            target_output[name]=original_output[name].clone()
                            if  'MaskEOT'  in self.opt.algorithm:
                                    # print(generated_rect)    
                                target_output[name][:,:,generated_rect[0,0]:generated_rect[0,0]+generated_rect[0,1],generated_rect[0,2]:generated_rect[0,2]+generated_rect[0,3]]=torchvision.transforms.Resize((generated_rect[0,1],generated_rect[0,3]))(target_patch)
                            else:
                            
                                target_patch_before_mask=torch.zeros(mask.size()[0],target_output[name].size()[1],mask.size()[-2],mask.size()[-1]).to(self.device)
                                target_patch_before_mask[:,:,rect[0,0,0]:rect[0,0,0]+rect[0,0,1],rect[0,0,2]:rect[0,0,2]+rect[ 0,0,3]]=torchvision.transforms.Resize((rect[0,0,1],rect[0,0,3]))(target_patch)
                                target_output[name]=target_output[name]*(1-torchvision.transforms.Resize((target_output[name].size()[-2],target_output[name].size()[-1]))(mask.expand(-1,target_output[name].size()[1],-1,-1)))\
                                                + torchvision.transforms.Resize((target_output[name].size()[-2],target_output[name].size()[-1]))(target_patch_before_mask*mask.expand(-1,target_output[name].size()[1],-1,-1))
                     
                        if 'optimal_mask_search' in self.opt.algorithm:
                            # generated_mask=self.ADMM(input,noise_func,mask,target_patch=original_output,original_FeatList=original_FeatList)
                            # assert cv2.imwrite('/home1/mingzhi/inpainting/ensemble/experiment_result/generated_mask.jpg',generated_mask.detach().cpu().numpy()[0].transpose(1,2,0)*255)

                            raise 'no implementation error'
                            continue
                        else:
                            if 'perceptual_loss' in self.opt.lossType:
                               
                                if 'MaskEOT' in self.opt.algorithm:
                                    perceptual_loss=self.get_perceputal_loss(target_output,tmp_output,noise_func,rect=generated_rect)
                                else:
                                    perceptual_loss=self.get_perceputal_loss(target_output,tmp_output,noise_func,rect=rect)

                                
                                    
                                loss+=perceptual_loss
                            if 'lowlevel_perceptual_loss' in self.opt.lossType:
                                target_output={}
            
                                perceptual_loss=self.get_lowlevel_perceputal_loss_inpainting_model(original_FeatList,tmp_FeatList,noise_func,rect=rect)
                                    
                                loss+=-perceptual_loss
                            
                            if 'output_pixel_loss' in self.opt.lossType:

                                pixel_loss=-50000*self.output_pixel_loss(target_output,tmp_output,mask)
                                loss+=pixel_loss
                            if 'CLIP' in self.opt.algorithm:
                                target_output={}
                            
                            if 'gradient_loss' in self.opt.lossType:
                                gradient_loss=self.img_gradient_loss(target_output,tmp_output,rect,mask)
                                loss+= gradient_loss*1000

                               
                    # untarget attack (common case)
                    else:
                        if 'optimal_mask_search' in self.opt.algorithm:
                            # generated_mask=self.ADMM(input,noise_func,mask,target_patch=original_output)
                            # assert cv2.imwrite('/home1/mingzhi/inpainting/ensemble/experiment_result/generated_mask.jpg',generated_mask.detach().cpu().numpy()[0].transpose(1,2,0)*255)
                            
                            raise 'no implementation error'
                            continue

                        # pure pixel-wise loss
                        if 'output_pixel_loss' in self.opt.lossType:
                            if 'MaskEOT'  in self.opt.algorithm:
                         
                                loss+=50000*self.output_pixel_loss(original_output,tmp_output,mask)
                            else:
                             
                                loss+=500000*self.output_pixel_loss(original_output,tmp_output,mask)
                        # pure perceptual loss without AGP      
                        if 'perceptual_loss' in self.opt.lossType and 'attribution_attack_for_multi_task3' not in self.opt.algorithm:

                            perceptual_loss=self.get_perceputal_loss(original_output,tmp_output,noise_func)

                                
                            loss+=perceptual_loss
                        
                        if 'lowlevel_perceptual_loss' in self.opt.lossType:
                            perceptual_loss=1000*self.get_lowlevel_perceputal_loss_inpainting_model(original_FeatList,tmp_FeatList,noise_func)
                                
                            loss+=-perceptual_loss*1000

                        if 'similarity_inverse_loss' in self.opt.lossType:
                            loss+=10*self.similarity_inverse_loss(original_FeatList,tmp_FeatList)
                        if 'gradient_loss' in self.opt.lossType:
                            gradient_loss=self.img_gradient_loss(original_output,tmp_output,rect,mask)
                            loss+=-gradient_loss*1000
                        if 'optimal_transformation_search' in self.opt.lossType:
                            pass
                        
                        # baselnie method for blind watermark remover
                        if 'DWV' in self.opt.lossType:
                            original_pure_imoutput = {key: value[0] for key, value in original_output.items()}
                            tmp_pure_imoutput = {key: value[0] for key, value in tmp_output.items()}
                            DWVloss = self.output_pixel_loss(original_pure_imoutput,tmp_pure_imoutput)
                            loss += DWVloss
                        if 'IWV' in self.opt.lossType:
                            original_pure_imoutput = {key: value[0] for key, value in original_output.items()}
                            original_pure_mask = {key: value[1] for key, value in original_output.items()}

                            tmp_pure_imoutput = {key: value[0] for key, value in tmp_output.items()}
                            tmp_pure_mask = {key: value[1] for key, value in original_output.items()}


                            IWVloss = -2*self.output_pixel_loss(input.detach(),tmp_pure_imoutput) + self.output_pixel_loss(torch.zeros(input.size()).cuda(), tmp_pure_mask)
                            loss += IWVloss

                        # AGP (the one proposed in the paper): use attributes on adversaral examples in each iterations
                        if 'attribution_attack_for_multi_task3' in self.opt.algorithm:

                            if torch.sum(torch.isnan(tmp_input[list(tmp_input.keys())[0]])):
                                print('there is something wrong for tmp_input')
                                exit()

                            # get the adversarial example
                            single_tmp_input = tmp_input[list(tmp_input.keys())[0]]
                            single_original_input = input.clone().detach()

                            # separately attack two types of watermark removers for one step
                            
                            AE_for_inpainting = self.simple_attack(single_tmp_input, mask,original_output, steps = 1, tmp_target_models={key:value for key, value in self.target_models.items() if key in self.inpainting_model_names }, clean_input= single_original_input.clone().detach(), rect = rect)
                            AE_for_inpainting = torch.from_numpy(AE_for_inpainting).cuda()

                            AE_for_wrm = self.simple_attack(single_tmp_input, mask,original_output, steps = 1, tmp_target_models={key:value for key, value in self.target_models.items() if key in self.watermark_remover_names }, iter_index=i, clean_input= single_original_input.clone().detach(), rect = rect )
                            AE_for_wrm = torch.from_numpy(AE_for_wrm).cuda()

                            
                            # get the attributes of the two types of watermark removers respectively
                            with torch.autograd.detect_anomaly():
                                IG_for_inpainting_AE, tmp_loss = self.integrated_gradient({'original_input':single_original_input.clone().detach()},mask,original_output,steps=30, mode = 'inpainting', baseline = AE_for_inpainting, original_input=single_original_input)

                            IG_for_watermark_remover_AE, tmp_loss = self.integrated_gradient({'original_input':single_original_input.clone().detach()},mask,original_output,steps=30, mode = 'watermark_removal', baseline = AE_for_wrm, original_input=single_original_input)

                            IG_for_inpainting_AE = torch.from_numpy(IG_for_inpainting_AE).cuda()
                            IG_for_watermark_remover_AE = torch.from_numpy(IG_for_watermark_remover_AE).cuda()
                            # assert cv2.imwrite('inpainting/ensemble/pure_IG_for_watermark_remover_st_AE_before_norm.png', (torch.abs(IG_for_watermark_remover_AE)/torch.max(torch.abs(IG_for_watermark_remover_AE)))[0].detach().cpu().numpy().transpose(1,2,0)*255)
                            # assert cv2.imwrite('inpainting/ensemble/pure_IG_for_inpainting_st_AE_before_norm.png', (torch.abs(IG_for_inpainting_AE)/torch.max(torch.abs(IG_for_inpainting_AE)))[0].detach().cpu().numpy().transpose(1,2,0)*255)
                            

                            

                            ######################### combine the adversarial noise from two types of watermark removers
                            IG_for_inpainting_AE = IG_for_inpainting_AE/torch.sum(torch.abs(IG_for_inpainting_AE))
                            IG_for_watermark_remover_AE = IG_for_watermark_remover_AE/torch.sum(torch.abs(IG_for_watermark_remover_AE))
                            
                            sum_of_IG_for_inpainting_and_wr_AE = ((torch.abs(IG_for_inpainting_AE)+torch.abs(IG_for_watermark_remover_AE)) == 0)*1 + ((torch.abs(IG_for_inpainting_AE)+torch.abs(IG_for_watermark_remover_AE))!=0 )*(torch.abs(IG_for_inpainting_AE)+torch.abs(IG_for_watermark_remover_AE))
                            IG_for_inpainting_AE_after_norm = torch.abs(IG_for_inpainting_AE)/(sum_of_IG_for_inpainting_and_wr_AE)
                            IG_for_watermark_remover_AE_after_norm = torch.abs(IG_for_watermark_remover_AE)/(sum_of_IG_for_inpainting_and_wr_AE)


                            # assert cv2.imwrite('inpainting/ensemble/intermediate_output/pure_IG_for_all_task_st_AE_before_norm'+str(i)+'.png', (torch.abs(IG_for_inpainting_AE_after_norm + IG_for_watermark_remover_AE_after_norm)/torch.max(torch.abs(IG_for_inpainting_AE_after_norm + IG_for_watermark_remover_AE_after_norm)))[0].detach().cpu().numpy().transpose(1,2,0)*255)

                            IG_for_watermark_remover = IG_for_watermark_remover_AE_after_norm
                            IG_for_inpainting = IG_for_inpainting_AE_after_norm

                            adv_noise_for_inpainting = AE_for_inpainting - single_original_input
                            adv_noise_for_wrm = AE_for_wrm - single_original_input

                            # assert cv2.imwrite('inpainting/ensemble/intermediate_output/attacked_result_of_AE_for_wrm'+str(i)+'.jpg', self.target_models['SLBRModel'](AE_for_wrm)[0][0].detach().cpu().numpy().transpose(1,2,0)*255)
                            
                            
                            noise_func.delta_x.data = (adv_noise_for_inpainting*IG_for_inpainting + adv_noise_for_wrm*IG_for_watermark_remover).detach()
                            ######################################################################

                            noise_func.delta_x.grad =torch.zeros(noise_func.delta_x.size()).cuda()


                            # for debugging to track nan result
                            if torch.sum(torch.isnan(noise_func.delta_x)):
                                print('there is somethign wrong in noise_func.delta_x', torch.sum(torch.isnan(noise_func.delta_x)), torch.sum(torch.abs(IG_for_inpainting_AE==0)*torch.isnan(noise_func.delta_x)), torch.sum(torch.abs(IG_for_watermark_remover_AE==0)*torch.isnan(noise_func.delta_x) ), torch.sum((IG_for_inpainting_AE==0)*(IG_for_watermark_remover_AE==0)*torch.isnan(noise_func.delta_x) ) )
                                print('adv_noise_for_inpainting', torch.sum(torch.isnan(adv_noise_for_inpainting)), 'adv_noise_for_wrm', torch.sum(torch.isnan(adv_noise_for_wrm)), 'IG_for_inpainting', torch.sum(torch.isnan(IG_for_inpainting)), 'IG_for_watermark_remover', torch.sum(torch.isnan(IG_for_watermark_remover)))
                                exit()
                    # TIG attack
                    if 'TIG' in self.opt.lossType:
                        if self.opt.target_attack:
                            IG,tmp_loss=self.integrated_gradient(tmp_input,mask,target_output,steps=10)
                        else:
                            IG,tmp_loss=self.integrated_gradient(tmp_input,mask,original_output,steps=10)

                        if noise_func.delta_x.grad==None:
                            noise_func.delta_x.grad=1/len(preprocessed_inputs_with_noise)*torch.from_numpy(IG).to(self.device)
                        else:
                            noise_func.delta_x.grad+=1/len(preprocessed_inputs_with_noise)*torch.from_numpy(IG).to(self.device)

                    # separate TIG attack
                    if 'TIG_sep' in self.opt.lossType:
                        if self.opt.target_attack:
                            IG,tmp_loss=self.integrated_gradient(tmp_input,mask,target_output,steps=10)
                        else:
                            IG,tmp_loss=self.integrated_gradient(tmp_input,mask,original_output,steps=30, mode = 'separate', index_img = index, index_step=i, original_input= original_input)

                     
                        if noise_func.delta_x.grad==None:
                            noise_func.delta_x.grad=1/len(preprocessed_inputs_with_noise)*torch.from_numpy(IG).to(self.device)
                        else:
                            noise_func.delta_x.grad+=1/len(preprocessed_inputs_with_noise)*torch.from_numpy(IG).to(self.device)

                    if 'EG' in self.opt.lossType:
                        if self.opt.target_attack:
                            EG,tmp_loss=self.expectation_gradient(tmp_input,mask,target_output,steps=10)
                        else:
                            EG,tmp_loss=self.expectation_gradient(tmp_input,mask,original_output,steps=10)

                        if noise_func.delta_x.grad==None:
                            noise_func.delta_x.grad=1/len(preprocessed_inputs_with_noise)*torch.from_numpy(EG).to(self.device)
                        else:
                            noise_func.delta_x.grad+=1/len(preprocessed_inputs_with_noise)*torch.from_numpy(EG).to(self.device)


                    
                    
                if 'optimal_mask_search' in self.opt.algorithm:
                    print('deltax ',torch.max(noise_func.delta_x))
                    continue
                

                print('loss',loss, self.opt.output_dir)

                ######################### for attribute-free attack method, it needs to cal gradient here
                if 'TIG' not in self.opt.lossType and 'TIG_sep' not in self.opt.lossType and 'EG' not in self.opt.lossType and 'optimal_mask_search' not in self.opt.algorithm and 'attribution_attack_for_multi_task3' not in self.opt.algorithm:
                    loss.backward()

                
                print('max grad before sign',torch.max(torch.abs(noise_func.delta_x.grad)))


                # normalize the adversarial noise correspondnig to their gradient on different models (no obvious change in performance)
                if 'norm_combine' in self.opt.algorithm and inter_grad!=None and 'TIG' not in self.opt.lossType and 'TIG_sep' not in self.opt.lossType and 'EG' not in self.opt.lossType and 'attribution_attack_for_multi_task' not in self.opt.algorithm and 'attribution_attack_for_multi_task2' not in self.opt.algorithm  and 'attribution_attack_for_multi_task3' not in self.opt.algorithm:
                    sum_grad=0
                    sum_grad_for_inpainting =0
                    sum_grad_for_watermark_removal = 0
                    for name in inter_grad['attack']['delta_x']:
                        sum_grad+=inter_grad['attack']['delta_x'][name]/torch.linalg.norm(inter_grad['attack']['delta_x'][name]) 
                        if name in self.inpainting_model_names:
                            sum_grad_for_inpainting+=inter_grad['attack']['delta_x'][name]/torch.linalg.norm(inter_grad['attack']['delta_x'][name])
                        if name in self.watermark_remover_names:
                            sum_grad_for_watermark_removal+=inter_grad['attack']['delta_x'][name]/torch.linalg.norm(inter_grad['attack']['delta_x'][name])
                    
                    noise_func.delta_x.grad.data=sum_grad.data 
                    
                    # save the perturbation for further view
                    if torch.is_tensor(sum_grad_for_inpainting):
                        with open(os.path.join('/home1/mingzhi/inpainting/ensemble/sep_perturbations/baseline_grad_sep', 'inpainting'+str(index)+'_'+str(i)+'.npy'), 'wb') as  f:
                            perturbation_for_inpainting = (sum_grad_for_inpainting.data/torch.norm(sum_grad_for_inpainting.data, p=1)).detach().cpu().numpy()

                            assert cv2.imwrite('inpainting/ensemble/inpainting_image.jpg', np.sign(np.abs(perturbation_for_inpainting)/np.max(np.abs(perturbation_for_inpainting)))[0].transpose(1,2,0)*255)

                            
                            
                            perturbation_for_inpainting = 1.0*(np.sign(1.0*(np.sign(perturbation_for_inpainting)<0)*((input-original_input).detach().cpu().numpy()>=self.budget)+(np.sign(perturbation_for_inpainting)>0)*((input-original_input).detach().cpu().numpy()<=-self.budget)) == 0)*perturbation_for_inpainting
                            np.save(f, perturbation_for_inpainting)
                            assert cv2.imwrite('inpainting/ensemble/inpainting_filtered_image.jpg', np.sign(np.abs(perturbation_for_inpainting)/np.max(np.abs(perturbation_for_inpainting)))[0].transpose(1,2,0)*255)
                            
                    
                    if torch.is_tensor(sum_grad_for_watermark_removal):
                        with open(os.path.join('/home1/mingzhi/inpainting/ensemble/sep_perturbations/baseline_grad_sep', 'bwr'+str(index)+'_'+str(i)+'.npy'), 'wb') as  f:
                            perturbation_for_bwr = (sum_grad_for_watermark_removal.data/torch.norm(sum_grad_for_watermark_removal.data, p=1)).detach().cpu().numpy()
                            perturbation_for_bwr = 1.0*(np.sign(1.0*(np.sign(perturbation_for_bwr)<0)*((input-original_input).detach().cpu().numpy()>=self.budget)+(np.sign(perturbation_for_bwr)>0)*((input-original_input).detach().cpu().numpy()<=-self.budget)) ==0 )*perturbation_for_bwr
                            np.save(f, perturbation_for_bwr)

                   
                    
                    
                    
                    
                # frist option for attribute-guided attack
                if 'attribution_attack_for_multi_task' in self.opt.algorithm:
                    sum_grad_for_inpainting = 0
                    for name in inter_grad['attack']['delta_x']:
                        if name in self.inpainting_model_names:
                            sum_grad_for_inpainting+=inter_grad['attack']['delta_x'][name]/torch.linalg.norm(inter_grad['attack']['delta_x'][name])
                    
                    sum_grad_for_watermark_removal = 0
                    for name in inter_grad['attack']['delta_x']:
                        if name in self.watermark_remover_names:
                            sum_grad_for_watermark_removal+=inter_grad['attack']['delta_x'][name]/torch.linalg.norm(inter_grad['attack']['delta_x'][name])

                    
                    
                    noise_func.delta_x.grad.data= (sum_grad_for_inpainting.data/torch.norm(sum_grad_for_inpainting.data, p=1))*IG_for_inpainting + (sum_grad_for_watermark_removal.data/torch.norm(sum_grad_for_watermark_removal.data, p=1)) * IG_for_watermark_remover
                
                if 'TI' in self.opt.algorithm:
                    noise_func.delta_x.grad=self.grad_of_preprocess(noise_func.delta_x.grad)

                # get the sign of the gradient to update adversarial noise
                if self.opt.sign:
                    if  'MaskEOT'  in self.opt.algorithm:
                        noise_func.delta_x.grad=random.random()*torch.sign(noise_func.delta_x.grad).to(self.device)
                    else:
                        noise_func.delta_x.grad=torch.sign(noise_func.delta_x.grad).to(self.device)
                print('max grad after sign',torch.max(torch.abs(noise_func.delta_x.grad)))
                print('max noise,', torch.max(noise_func.delta_x))
                

                # for the other methods than AGP, update the adversarial noise with their gradient here
                if 'attribution_attack_for_multi_task3' not in self.opt.algorithm:
                    attack_optimizer.step()
                    scheduler.step()

                noise_func.delta_x.data=torch.clamp((torch.clamp(noise_func.delta_x.detach()+input,min=0,max=1)-input),min=-self.budget,max=self.budget).detach()
            noise_func.delta_x.data=torch.clamp((torch.clamp(noise_func.delta_x.detach()+input,min=0,max=1)-input),min=-self.budget,max=self.budget).detach()
            input_with_final_noise=noise_func(input.detach().clone())
        
        # the second version of attribute guided attack (not the one proposed in paper): use one step AGP 
        if 'attribution_attack_for_multi_task2' in self.opt.algorithm:
            


            AE_for_inpainting = self.simple_attack(input, mask,original_output, steps = 100, tmp_target_models={key:value for key, value in self.target_models.items() if key in self.inpainting_model_names })
            AE_for_inpainting = torch.from_numpy(AE_for_inpainting).cuda()

            AE_for_wrm = self.simple_attack(input, mask,original_output, steps = 100, tmp_target_models={key:value for key, value in self.target_models.items() if key in self.watermark_remover_names })
            AE_for_wrm = torch.from_numpy(AE_for_wrm).cuda()

            IG_for_inpainting_AE, tmp_loss = self.integrated_gradient({'original_input':input.clone().detach()},mask,original_output,steps=30, mode = 'inpainting', baseline = AE_for_inpainting)

            IG_for_watermark_remover_AE, tmp_loss = self.integrated_gradient({'original_input':input.clone().detach()},mask,original_output,steps=30, mode = 'watermark_removal', baseline = AE_for_wrm)

            IG_for_inpainting_AE = torch.from_numpy(IG_for_inpainting_AE).cuda()
            IG_for_watermark_remover_AE = torch.from_numpy(IG_for_watermark_remover_AE).cuda()
            assert cv2.imwrite('inpainting/ensemble/pure_IG_for_watermark_remover_st_AE_before_norm.png', (torch.abs(IG_for_watermark_remover_AE)/torch.max(torch.abs(IG_for_watermark_remover_AE)))[0].detach().cpu().numpy().transpose(1,2,0)*255)
            assert cv2.imwrite('inpainting/ensemble/pure_IG_for_inpainting_st_AE_before_norm.png', (torch.abs(IG_for_inpainting_AE)/torch.max(torch.abs(IG_for_inpainting_AE)))[0].detach().cpu().numpy().transpose(1,2,0)*255)
           
            IG_for_inpainting_AE = IG_for_inpainting_AE/torch.sum(torch.abs(IG_for_inpainting_AE))
            IG_for_watermark_remover_AE = IG_for_watermark_remover_AE/torch.sum(torch.abs(IG_for_watermark_remover_AE))
            
            
            IG_for_inpainting_AE_after_norm = torch.abs(IG_for_inpainting_AE)/(torch.abs(IG_for_inpainting_AE)+torch.abs(IG_for_watermark_remover_AE))
            IG_for_watermark_remover_AE_after_norm = torch.abs(IG_for_watermark_remover_AE)/(torch.abs(IG_for_inpainting_AE)+torch.abs(IG_for_watermark_remover_AE))


            IG_for_watermark_remover = IG_for_watermark_remover_AE_after_norm
            IG_for_inpainting = IG_for_inpainting_AE_after_norm

            adv_noise_for_inpainting = AE_for_inpainting - input
            adv_noise_for_wrm = AE_for_wrm - input
            for name in input_with_final_noise:
                input_with_final_noise[name] = input.clone().detach() + adv_noise_for_inpainting*IG_for_inpainting + adv_noise_for_wrm*IG_for_watermark_remover
     


        # get the evaluation outputs from test model
        evaluate_output={}
        for name in self.target_models.keys():

            # it is usually this case
            if 'optimal_mask_search' not in self.opt.algorithm:
                if name in [ 'WDModel', 'DBWEModel', 'SLBRModel']:
                    evaluate_output[name]=self.target_models[name](input_with_final_noise[name])
                else:
                    evaluate_output[name]=self.target_models[name](input_with_final_noise[name],original_mask.detach().clone())
            # no implementation
            else:
                evaluate_output[name]=self.target_models[name](input*(1-generated_mask),generated_mask.detach().clone())
         
        # get the validation output form the validation model
        validation_output={}
        original_output_for_validation={}
        validation_value=0
        for name in self.validation_models:
            original_output_for_validation[name]=self.validation_models[name](input,original_mask.detach().clone()).clone().detach()
            validation_output[name]=self.validation_models[name](list(input_with_final_noise.values())[0],original_mask.detach().clone()).clone().detach()
            validation_value+=self.get_loss(original_output=original_output_for_validation,tmp_output=validation_output,mask=original_mask).item()
            original_output_for_validation[name]=self.validation_models[name](input,original_mask.detach().clone()).clone().detach()
            validation_output[name]=validation_output[name].cpu()
            original_output_for_validation[name] = original_output_for_validation[name].cpu()
        

        # no implementation
        if 'optimal_mask_search' in self.opt.algorithm:
            return input_with_final_noise[list(self.target_models.keys())[0]], (input_with_final_noise[list(self.target_models.keys())[0]]-original_input).detach().cpu(), evaluate_output, generated_mask.detach()

        
        return input_with_final_noise[list(self.target_models.keys())[0]], (input_with_final_noise[list(self.target_models.keys())[0]]-original_input).detach().cpu(), evaluate_output,mask.detach().cpu(),validation_output,original_output_for_validation,validation_value
        # return input_with_final_noise.detach().cpu(), (input_with_final_noise-original_input).detach().cpu(), evaluate_output

    # to get the test result by following the template of attack
    def evaluate(self,input,attacked_input,original_output,original_mask,target_patch=None,rect=None,original_FeatList=None):
        
        assert torch.is_tensor(input) and torch.is_tensor(original_mask)
        assert input.dim()==4 and original_mask.dim()==4
        input=input.cuda()+torch.rand(input.size()).cuda()*0.02

        if 'recover_attack' in self.opt.algorithm:
            blank_input=torch.zeros(input.size()).cuda()
            
        original_mask=original_mask.cuda()
        original_input=input.clone().detach()

        if self.opt.target_attack:
            assert torch.is_tensor(target_patch) and target_patch.dim()==4


        input.requires_grad=True
        noise_func=Noise_Func(input.size(),None,None,target_models=self.target_models,PGD=False).cuda()

        # set the adversarial noise
        if attacked_input!=None:
            noise_func.delta_x=attacked_input-input
        

        loss=0


        with torch.enable_grad():

            # for i in range(self.iters):
            for i in range(1):
                

                input_with_noise=noise_func(input.clone())
                preprocessed_inputs_with_noise=[input_with_noise]

                

                if 'MaskEOT'  in self.opt.algorithm:
                    mask, generated_rect = generate_rect_mask((int(input.size()[-2] ),int(input.size()[-1] )), (int(input.size()[-2]*0.5),int(input.size()[-1]*0.5)), rand_mask=True,batch_size=input.size()[0])
                    mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()

                else:
                    mask= original_mask.clone().detach()
                
                # get the original output of the target model and the loss of attack
                for tmp_input in preprocessed_inputs_with_noise:
                    tmp_output={}
                    tmp_FeatList={}

                
                    for name,_ in self.target_models.items():
                        if 'lowlevel_perceptual_loss' not in self.opt.lossType and   'similarity_inverse_loss' not in self.opt.lossType:

                            
                            tmp_output[name]=self.target_models[name](tmp_input[name],mask)

                        if 'lowlevel_perceptual_loss'   in self.opt.lossType or 'similarity_inverse_loss' in self.opt.lossType:

                            tmp_output[name],tmp_FeatList[name]=self.target_models[name](tmp_input[name],mask,keepFeat=[0,1,2,3])

                    # not common cases    
                    if self.opt.target_attack:
                        target_output={}
                        for name in tmp_output.keys():
                            
                            target_output[name]=original_output[name].clone()
                            if  'MaskEOT'  in self.opt.algorithm:
                                target_output[name][:,:,generated_rect[0,0]:generated_rect[0,0]+generated_rect[0,1],generated_rect[0,2]:generated_rect[0,2]+generated_rect[0,3]]=torchvision.transforms.Resize((generated_rect[0,1],generated_rect[0,3]))(target_patch)
                            else:
        
                                target_patch_before_mask=torch.zeros(mask.size()[0],target_output[name].size()[1],mask.size()[-2],mask.size()[-1]).to(self.device)
                                target_patch_before_mask[:,:,rect[0,0,0]:rect[0,0,0]+rect[0,0,1],rect[0,0,2]:rect[0,0,2]+rect[ 0,0,3]]=torchvision.transforms.Resize((rect[0,0,1],rect[0,0,3]))(target_patch)
                                target_output[name]=target_output[name]*(1-torchvision.transforms.Resize((target_output[name].size()[-2],target_output[name].size()[-1]))(mask.expand(-1,target_output[name].size()[1],-1,-1)))\
                                                + torchvision.transforms.Resize((target_output[name].size()[-2],target_output[name].size()[-1]))(target_patch_before_mask*mask.expand(-1,target_output[name].size()[1],-1,-1))
 
                        if 'optimal_mask_search' in self.opt.algorithm:
                            generated_mask=self.ADMM(input,noise_func,mask,target_patch=original_output,original_FeatList=original_FeatList)
    
                            continue
                        else:
                            if 'perceptual_loss' in self.opt.lossType:

                                if 'MaskEOT' in self.opt.algorithm:
                                    perceptual_loss=self.get_perceputal_loss(target_output,tmp_output,noise_func,rect=generated_rect)
                                else:
                                    perceptual_loss=self.get_perceputal_loss(target_output,tmp_output,noise_func,rect=rect)

                                
                                    
                                loss+=perceptual_loss
                            if 'lowlevel_perceptual_loss' in self.opt.lossType:
                                target_output={}
                              
                                perceptual_loss=self.get_lowlevel_perceputal_loss_inpainting_model(original_FeatList,tmp_FeatList,noise_func,rect=rect)
                                    
                                loss+=-perceptual_loss
                            
                            if 'output_pixel_loss' in self.opt.lossType:
                           
                                pixel_loss=-50000*self.output_pixel_loss(target_output,tmp_output,mask)
                                loss+=pixel_loss
                            if 'CLIP' in self.opt.algorithm:
                                target_output={}
                            
                            if 'gradient_loss' in self.opt.lossType:
                                gradient_loss=self.img_gradient_loss(target_output,tmp_output,rect,mask)
                                loss+= gradient_loss*1000

                               
                    # common case
                    else:
                        # no implementation
                        if 'optimal_mask_search' in self.opt.algorithm:
                            generated_mask=self.ADMM(input,noise_func,mask,target_patch=original_output)
                            assert cv2.imwrite('/home1/mingzhi/inpainting/ensemble/experiment_result/generated_mask.jpg',generated_mask.detach().cpu().numpy()[0].transpose(1,2,0)*255)
                            
                            continue
                        if 'output_pixel_loss' in self.opt.lossType:
                            if 'MaskEOT'  in self.opt.algorithm:
                                loss+=50000*self.output_pixel_loss(original_output,tmp_output,mask)
                            else:
                                loss+=500000*self.output_pixel_loss(original_output,tmp_output,mask)
                        if 'perceptual_loss' in self.opt.lossType:
                            
                            perceptual_loss=self.get_perceputal_loss(original_output,tmp_output,noise_func)
                                
                            loss+=-perceptual_loss
                        
                        if 'lowlevel_perceptual_loss' in self.opt.lossType:

                            perceptual_loss=1000*self.get_lowlevel_perceputal_loss_inpainting_model(original_FeatList,tmp_FeatList,noise_func)
                                
                            loss+=-perceptual_loss*1000
                        if 'similarity_inverse_loss' in self.opt.lossType:
                            loss+=10*self.similarity_inverse_loss(original_FeatList,tmp_FeatList)
                        if 'gradient_loss' in self.opt.lossType:
                            gradient_loss=self.img_gradient_loss(original_output,tmp_output,rect,mask)
                            loss+=-gradient_loss*1000
                        if 'optimal_transformation_search' in self.opt.lossType:
                            pass
                        
                    if 'TIG' in self.opt.lossType:
                        if self.opt.target_attack:
                            IG,tmp_loss=self.integrated_gradient(tmp_input,mask,target_output,steps=10)
                        else:
                            IG,tmp_loss=self.integrated_gradient(tmp_input,mask,original_output,steps=10)




                    
                    
                if 'optimal_mask_search' in self.opt.algorithm:
                    print('deltax ',torch.max(noise_func.delta_x))
                    continue
                

                print('loss',loss, self.opt.output_dir)


        input_with_final_noise=noise_func(input.detach().clone())

        # get the test output of the adversarial examples
        evaluate_output={}
        for name in self.target_models.keys():
            if 'optimal_mask_search' not in self.opt.algorithm:
                evaluate_output[name]=self.target_models[name](input_with_final_noise[name],original_mask.detach().clone())
            else:
                evaluate_output[name]=self.target_models[name](input*(1-generated_mask),generated_mask.detach().clone())

        return evaluate_output, loss.item()

    def grad_of_preprocess(self,grad):
       
        shift=nn.Conv2d(grad.size()[1],grad.size()[1],kernel_size=13,stride=1,padding=6,bias=False).to(self.device)
        
        kernel=torch.zeros((shift.weight.data.size()))

        kernel=torch.zeros((shift.weight.data.size()))
        # kernelB=torch.zeros((shiftB.weight.data.size()))
        center_x=int((kernel.size()[-2]-1)/2)
        center_y=int((kernel.size()[-1]-1)/2)
        for i in range(grad.size()[1]):
            kernel[i,i,:,:]=torch.ones((kernel.size()[-2],kernel.size()[-1]))/torch.sum(torch.ones((kernel.size()[-2],kernel.size()[-1])))
            #################################
            
 
            kernel[i,i,:,:]=kernel[i,i,:,:]/torch.sum(kernel[i,i,:,:])
        shift.weight.data=kernel.to(self.device)
        resizes=[]
 
        normalizes=[]
        for i in range(100):
            normalizes.append(0.5+(random.random()-0.5)*0.1)
        grad=shift(grad)
     
        grad_sum=0
        for denormalize in normalizes:
            grad_sum+=grad/denormalize
        grad=grad_sum/len(normalizes)
        return grad
        pass 

    def expectation_gradient(self,input,mask,y,steps,tau=0.1):
        # input=input.detach()
        mask=mask.detach()
        
        original_outputs={}
        for name in self.target_models.keys():
            original_outputs[name]=y[name].clone().detach()
        gradients=[]
        # baseline=torch.rand(input[list(input.keys())[0]].size()).to(self.device)
        baseline=torch.zeros(input[list(input.keys())[0]].size()).to(self.device)
        # scaled_inputs = [baseline + (float(i) / steps) * (input - baseline)+(torch.rand(baseline.size())-0.5).to(self.device)*tau for i in
        #              range(0, steps + 1)]
 
        
        
        grads=[]
        final_loss=0
        baseline_loss=0
        original_loss=0
        # for index,scaled_input in enumerate(scaled_inputs):
        for index in range(steps+1):
            loss=0
            tmp_outputs={}
            # scaled_input=scaled_input.clone().detach()
            # tgt=tgt.clone().detach()
            
            name = list(input.keys())[0]
            # scaled_input=(input[name].detach()-((index/steps)*(input[name]-baseline))).detach() + 0.01*torch.rand(input[list(input.keys())[0]].size()).to(self.device)
            # scaled_input=(baseline + ((index/steps)*(input[name]-baseline))).detach() + 0.001*torch.rand(input[list(input.keys())[0]].size()).to(self.device)
            scaled_input=input[name].detach() + (index/steps)*torch.rand(input[list(input.keys())[0]].size()).to(self.device)
            
            inter_grad = {}
            def print_grad(name):
                if inter_grad is None:
                    # return lambda grad: print('biggest grad for '+str(name)+' is ',torch.max(torch.abs(grad)))
                    def grad_func(grad):
                        pass
                    return grad_func
                else:
                    def grad_func(grad):
                        inter_grad[name]=grad
                    return grad_func
                
            # scaled_input.requires_grad=True
            # scaled_inputs={}
            
            # for name in self.target_models.keys():
            #     scaled_inputs[name]=scaled_input.clone()


            for name in self.target_models.keys():
                # y[name]=y[name].clone().detach()
                self.target_models[name].zero_grad()

            cloned_scaled_inputs={}
            for name in self.target_models.keys():                
                
                cloned_scaled_inputs[name] = scaled_input.clone().detach()
                cloned_scaled_inputs[name].requires_grad=True
                cloned_scaled_inputs[name].register_hook(print_grad(name))
                
                cloned_scaled_inputs[name].grad=None
                
                assert cloned_scaled_inputs[name].dim()==4 and mask.dim()==4
                # print(scaled_input.size(),tgt.size())
                tmp_output=self.target_models[name](cloned_scaled_inputs[name],mask)
                # tmp_output=self.model[name].depreprocess(tmp_output)
                assert tmp_output.dim()==4
                tmp_outputs[name]=tmp_output
            
            # loss+=20000*self.ensemble_face_recognition_loss(tmp_outputs,y)
            # assert cv2.imwrite('inpainting/ensemble/original_output.jpg',original_outputs[list(original_outputs.keys())[0]][0].detach().cpu().numpy().transpose(1,2,0)*255)
            # exit()
            if self.opt.target_attack:   
                loss+=-50000*self.get_perceputal_loss(original_outputs,tmp_outputs,mask)
            else:
                loss+=50000*self.get_perceputal_loss(original_outputs,tmp_outputs,mask)
            print('loss in EG', loss.item())
            loss.backward()
            # loss=loss.detach()
            # print(scaled_input.grad.detach())
            # exit()
            # grads.append(scaled_input.grad.detach()) 
            print(inter_grad.keys())
            
            sum_grad = 0
            for name in self.target_models:
                sum_grad+=inter_grad[name]/torch.linalg.norm(inter_grad[name])
            grads.append(sum_grad.clone().detach())
            final_loss=loss.detach().data

            del scaled_input, tmp_outputs,loss, inter_grad, cloned_scaled_inputs
            # del scaled_input,loss
            # for name in self.target_models:
            #     Path('inpainting/ensemble/IG_attribute_'+name).mkdir(parents=True,exist_ok=True)
    
            #     assert cv2.imwrite('inpainting/ensemble/IG_attribute_'+name+'/IG_final_output_'+name+str(index)+'.jpg',tmp_outputs[name][0].detach().cpu().numpy().transpose(1,2,0)*255)

        # print(grads)
        # exit()
        avg_grads = torch.mean(torch.cat(grads,0), dim=0, keepdim=True)
        
        # delta_X = scaled_inputs[-1] - scaled_inputs[0]
        delta_X=input[list(input.keys())[0]].detach() - baseline
        expectation_grads = avg_grads * torch.sign(delta_X)
        EG=expectation_grads.detach().cpu().numpy()
        
        # print(
        #     IG.shape)
        # exit(0)
        # print()
        
        # assert cv2.imwrite('inpainting/ensemble/input_example.jpg',input[list(input.keys())[0]][0].detach().cpu().numpy().transpose(1,2,0)*255)
        assert cv2.imwrite('inpainting/ensemble/EG.jpg',(np.abs(EG)/np.max(np.abs(EG)))[0].transpose(1,2,0)*255)
        # exit()
        del delta_X, avg_grads,input,original_outputs,baseline,grads,expectation_grads
        
        if self.opt.target_attack:
            return  EG,final_loss.cpu().detach().numpy()
        return  EG,final_loss.cpu().detach().numpy()


    # calculate IG
    def integrated_gradient(self,input,mask,y,steps,tau=0.1, mode = 'all', baseline = None, index_img = None, index_step = None, original_input = None ):
        # input=input.detach()
        original_input = original_input.detach().cpu().numpy()
        mask=mask.detach()
        
        # get original output
        original_outputs={}
        for name in self.target_models.keys():
            if mode == 'inpainting':
                if name not in self.inpainting_model_names:
                    continue
            elif mode == 'watermark_removal':
                if name not in self.watermark_remover_names:
                    continue
            if name in self.watermark_remover_names:
                original_outputs[name]=y[name][0].clone().detach()

            else:
                original_outputs[name]=y[name].clone().detach()
        gradients=[]
        if baseline == None:
            baseline=torch.zeros(input[list(input.keys())[0]].size()).to(self.device)
   
        
        
        grads=[]
        final_loss=0
        baseline_loss=0
        original_loss=0
        for index in range(steps+1):
            loss=0
            tmp_outputs={}
  
            
            name = list(input.keys())[0]

            # The index-th intermediate point
            scaled_input=torch.clamp((baseline + ((index/steps)*(input[name]-baseline))).detach() + 0.03*2*(torch.rand(input[list(input.keys())[0]].size())-0.5).to(self.device), min =0, max =1)
            scaled_input.requires_grad=True

            
            # zero-gradient operation on the target model
            for name in self.target_models.keys():
                if mode == 'inpainting':
                    if name not in self.inpainting_model_names:
                        continue
                elif mode == 'watermark_removal':
                    if name not in self.watermark_remover_names:
                        continue
                self.target_models[name].zero_grad()

            inter_grad = {}
            # save gradient
            def print_grad(name, mode = 'input'):
                if inter_grad is None:
                    def grad_func(grad):
                        pass
                    return grad_func
                else:
                    def grad_func(grad):

                        if torch.sum(torch.isnan(grad)) > 0:
                            print(name, mode, 'has something wrong in grad')
                        else:
                            # print(name, mode, 'has nothing wrong in grad', torch.max(torch.abs(grad)))
                            pass

                        inter_grad[name]=grad
                    return grad_func

            # register hood for each model
            cloned_scaled_inputs={}
            for name in self.target_models.keys():                
                if mode == 'inpainting':
                    if name not in self.inpainting_model_names:
                        continue
                elif mode == 'watermark_removal':
                    if name not in self.watermark_remover_names:
                        continue
                scaled_input.grad=None
                    
                cloned_scaled_inputs[name] = scaled_input.clone().detach()
                cloned_scaled_inputs[name].requires_grad=True
                cloned_scaled_inputs[name].register_hook(print_grad(name))
                
                cloned_scaled_inputs[name].grad=None
                
                assert cloned_scaled_inputs[name].dim()==4 and mask.dim()==4
                
                # get the output for each model
                if name in self.watermark_remover_names:
                    tmp_output=self.target_models[name](cloned_scaled_inputs[name])
                    tmp_output=tmp_output[2]
                else:
                    # tmp_output=self.target_models[name](cloned_scaled_inputs[name],mask, register_hook=True)
                    tmp_output=self.target_models[name](cloned_scaled_inputs[name],mask)
                    tmp_output=tmp_output
                    # tmp_output.register_hook(print_grad(name, mode = 'output'))
                assert tmp_output.dim()==4
                tmp_outputs[name]=tmp_output


            if self.opt.target_attack:   
                loss+=-50000*self.get_perceputal_loss(original_outputs,tmp_outputs,mask)
            else:
                loss+=50000*self.get_perceputal_loss(original_outputs,tmp_outputs,mask)
            loss.backward()

            # get all gradient from all the models
            if mode != 'separate':
                sum_grad = 0
                for name in self.target_models:
                    if mode == 'inpainting':
                        if name not in self.inpainting_model_names:
                            continue
                    elif mode == 'watermark_removal':
                        if name not in self.watermark_remover_names:
                            continue
                    
                    if torch.sum(torch.isnan(inter_grad[name])):
                        print(inter_grad[name])
                        print('something wrong with the integrated calculation', name, 'loss', torch.isnan(loss), loss, 'original_outputs', torch.sum(torch.isnan(original_outputs[name])), 'tmp_outputs', torch.sum(torch.isnan(tmp_outputs[name])), 'input', torch.sum(torch.isnan(cloned_scaled_inputs[name])), 'mask', torch.sum(torch.isnan(mask)))

                        assert cv2.imwrite('inpainting/ensemble/problem_output.jpg', tmp_outputs[name].detach().cpu()[0].numpy()[::-1].transpose(1,2,0) * 255)
                        

                    sum_grad+=inter_grad[name]
                grads.append(sum_grad.clone().detach())
                for name in self.target_models:
                    if mode == 'inpainting':
                        if name not in self.inpainting_model_names:
                            continue
                    elif mode == 'watermark_removal':
                        if name not in self.watermark_remover_names:
                            continue
                    if torch.sum(torch.isnan(inter_grad[name])):
                        exit()
            # get gradient from different types of models separately
            elif mode == 'separate':

                sum_grad_for_wr = {}
                for name in self.target_models:
                    if name not in self.watermark_remover_names:
                        continue
                    sum_grad_for_wr[name] = inter_grad[name].clone().detach()
                sum_grad_for_inpainting = {}
                for name in self.target_models:
                    if name not in self.inpainting_model_names:
                        continue
                    sum_grad_for_inpainting[name] = inter_grad[name].clone().detach()
                grads.append({'inpainting':sum_grad_for_inpainting,'wr':sum_grad_for_wr})
            
    
            final_loss=loss.clone().detach().data.cpu().numpy()

            del scaled_input, tmp_outputs,loss, inter_grad, cloned_scaled_inputs
            
   
        # use gradient of all the models to cal IG
        if mode != 'separate':
            avg_grads = torch.mean(torch.cat(grads,0), dim=0, keepdim=True)
        
            delta_X=input[list(input.keys())[0]].detach() - baseline
            integrated_grad = avg_grads * delta_X
            IG=integrated_grad.detach().cpu().numpy()
            del delta_X, avg_grads,input,original_outputs,baseline,grads,integrated_grad
        
        # combine IG by separately normalize the IGs from different types of watermark removers separately
        elif mode == 'separate':
            mean_grad_for_inpainting = {}
            for name in self.target_models:
                if name not in self.inpainting_model_names:
                    continue
                mean_grad_for_inpainting[name] = [grad['inpainting'][name] for grad in grads]
                mean_grad_for_inpainting[name]=torch.mean(torch.cat(mean_grad_for_inpainting[name], dim = 0), dim =0, keepdim=True)
                mean_grad_for_inpainting[name]=mean_grad_for_inpainting[name]/torch.norm(mean_grad_for_inpainting[name],p=1)
            
            # get the mean of normalized mean gradient of each inpainting model
            if mean_grad_for_inpainting != {}:
                avg_grads_for_inpainting = torch.mean(torch.cat([grad for key, grad in mean_grad_for_inpainting.items()],0), dim=0, keepdim=True)

                global figure_for_paper_index
                """ 
                # avg_grads_for_inpainting_grey = avg_grads_for_inpainting
                avg_grads_for_inpainting_grey = torch.sum(avg_grads_for_inpainting,dim=1, keepdim=True)
                inpainting_grey_numpy =(avg_grads_for_inpainting_grey/torch.norm(avg_grads_for_inpainting_grey, p=1)).detach().cpu().numpy()
                inpainting_grey_numpy = 1.0*((1.0*(np.sign(inpainting_grey_numpy)<0)*((input[list(input.keys())[0]].detach().cpu().numpy()-original_input)>=self.budget)+(np.sign(inpainting_grey_numpy)>0)*((input[list(input.keys())[0]].detach().cpu().numpy()-original_input)<=-self.budget)) == 0)*inpainting_grey_numpy

                assert cv2.imwrite('inpainting/ensemble/inpainting_grey_numpy.jpg', np.sign(np.abs(inpainting_grey_numpy))[0].transpose(1,2,0)*255)
                with open(os.path.join('inpainting/ensemble/sep_perturbations/', 'ours_TIG_sep','inpainting'+str(index_img)+'_'+str(index_step)+'.npy'), 'wb') as f:
                    np.save(f, inpainting_grey_numpy)

                del avg_grads_for_inpainting_grey """
            else:
                avg_grads_for_inpainting = 0
        
            mean_grad_for_wr = {}
            for name in self.target_models:
                if name not in self.watermark_remover_names:
                    continue
                mean_grad_for_wr[name] = [grad['wr'][name] for grad in grads]
                mean_grad_for_wr[name]=torch.mean(torch.cat(mean_grad_for_wr[name], dim = 0), dim =0, keepdim=True)
                mean_grad_for_wr[name]=mean_grad_for_wr[name]/torch.norm(mean_grad_for_wr[name],p=1)
            
            # get the mean of normalized mean gradient of each blind watermark model
            if mean_grad_for_wr!={}:
                avg_grads_for_wr = torch.mean(torch.cat([grad for key, grad in mean_grad_for_wr.items()],0), dim=0, keepdim=True)

                """ avg_grads_for_wr_grey = torch.sum(avg_grads_for_wr,dim=1, keepdim=True)
                bwr_grey_numpy =(avg_grads_for_wr_grey/torch.norm(avg_grads_for_wr_grey, p=1)).detach().cpu().numpy()
                bwr_grey_numpy = 1.0*((1.0*(np.sign(bwr_grey_numpy)<0)*((input[list(input.keys())[0]].detach().cpu().numpy()-original_input)>=self.budget)+(np.sign(bwr_grey_numpy)>0)*((input[list(input.keys())[0]].detach().cpu().numpy()-original_input)<=-self.budget)) == 0)*bwr_grey_numpy

                assert cv2.imwrite('inpainting/ensemble/bwr_grey_numpy.jpg', np.sign(np.abs(bwr_grey_numpy))[0].transpose(1,2,0)*255)
                with open(os.path.join('inpainting/ensemble/sep_perturbations/', 'ours_TIG_sep','bwr'+str(index_img)+'_'+str(index_step)+'.npy'), 'wb') as f:
                    np.save(f, bwr_grey_numpy)
                
                del avg_grads_for_wr_grey """
            else:
                avg_grads_for_wr = 0

            # if index_step>=29:
            #     exit()
            # avg_grads_for_wr_grey = 2*(torch.abs(avg_grads_for_wr_grey)/torch.max(torch.abs(avg_grads_for_wr_grey)))[0].detach().cpu().numpy().transpose(1,2,0)*255
            # avg_grads_for_wr_grey = cv2.applyColorMap(avg_grads_for_wr_grey.astype(np.uint8), cv2.COLORMAP_MAGMA)
            # assert cv2.imwrite('inpainting/ensemble/RIG_figure_for_paper/mean_grad_for_wr_for_paper'+str(figure_for_paper_index)+'.jpg', avg_grads_for_wr_grey)

            figure_for_paper_index+=1
            """  avg_grads_for_inpainting = torch.mean(torch.cat([grad['inpainting'] for grad in grads],0), dim=0, keepdim=True)
            avg_grads_for_wr = torch.mean(torch.cat([grad['wr'] for grad in grads],0), dim=0, keepdim=True) """
        
            # delta_X = scaled_inputs[-1] - scaled_inputs[0]
            delta_X=input[list(input.keys())[0]].detach() - baseline
            integrated_grad_for_inpainting = avg_grads_for_inpainting * delta_X
            integrated_grad_for_wr = avg_grads_for_wr * delta_X

            IG = (integrated_grad_for_inpainting/torch.norm(integrated_grad_for_inpainting, p=1) + integrated_grad_for_wr/torch.norm(integrated_grad_for_wr, p=1)).detach().cpu().numpy()
            # IG=integrated_grad.detach().cpu().numpy()
            del delta_X,input,original_outputs,baseline,grads,integrated_grad_for_inpainting, integrated_grad_for_wr, avg_grads_for_inpainting, avg_grads_for_wr, 
            
        
        # assert cv2.imwrite('inpainting/ensemble/input_example.jpg',input[list(input.keys())[0]][0].detach().cpu().numpy().transpose(1,2,0)*255)
        # assert cv2.imwrite('inpainting/ensemble/IG_'+mode+'.jpg',np.abs(np.abs(IG)/np.max(np.abs(IG)))[0].transpose(1,2,0)*255)
        # exit()
        
        
        if self.opt.target_attack:
            return  IG,final_loss
        return  IG,final_loss
    
    # standard attack with perceptual loss
    def simple_attack(self,input,mask,y,steps,tmp_target_models = None, iter_index = None, clean_input = None, rect = None):
        input=input.clone().detach()
        mask=mask.clone().detach()

        ################### get ready for attack
        if 'norm_combine' in self.opt.algorithm:
            inter_grad={}
        else:
            inter_grad=None
        noise_func=Noise_Func(clean_input.size(),None,None,target_models=tmp_target_models).cuda()
        parameter_to_learn=[{'params':noise_func.delta_x,'lr':self.lr}]
        attack_optimizer=optim.SGD(parameter_to_learn)
        scheduler=StepLR(attack_optimizer,step_size=50,gamma=0.5)

        noise_func.delta_x.data = (input - clean_input).detach()
        original_outputs={}
        for name in tmp_target_models.keys():
            if name in self.watermark_remover_names:
                original_outputs[name]=y[name][0].clone().detach()
            else:
                original_outputs[name]=y[name].clone().detach() 

        
        
        grads=[]
        final_loss=0
        baseline_loss=0
        original_loss=0
        with torch.enable_grad():
            for index in range(steps+1):
                loss=0
                
                attack_optimizer.zero_grad()
                clean_input.requires_grad=True

                input_with_noise=noise_func(clean_input.clone(),inter_grad=inter_grad)

                for name, fs_model in tmp_target_models.items():
                    tmp_target_models[name].zero_grad()
                

                # get the current output with adversarial examples
                tmp_outputs={}

                for name,_ in tmp_target_models.items():
                    if name in [ 'WDModel', 'DBWEModel', 'SLBRModel']:
                        tmp_outputs[name]=tmp_target_models[name](input_with_noise[name])
                    else:
                        tmp_outputs[name]=tmp_target_models[name](input_with_noise[name],mask)

                # cal loss
                if self.opt.target_attack:   
                    loss+=-50000*self.get_perceputal_loss(original_outputs,tmp_outputs, rect=rect)
                else:
                    loss+=50000*self.get_perceputal_loss(original_outputs,tmp_outputs, rect=None)
                loss.backward()
   
                
                sum_grad = 0    
                for name in tmp_target_models:
                    sum_grad+=inter_grad['attack']['delta_x'][name]/torch.linalg.norm(inter_grad['attack']['delta_x'][name])
                
                sum_grad = torch.sign(sum_grad)
                noise_func.delta_x.grad.data= sum_grad.detach()

                attack_optimizer.step()
                scheduler.step()

                # get the adversarial noise
                noise_func.delta_x.data=torch.clamp((torch.clamp(noise_func.delta_x.detach()+clean_input,min=0,max=1)-clean_input),min=-self.budget,max=self.budget).detach()

        

                if 'SLBRModel' not in tmp_target_models:
                    del sum_grad, loss, tmp_outputs

        if 'SLBRModel' in tmp_target_models:
            assert cv2.imwrite('inpainting/ensemble/intermediate_output/wr'+str(iter_index)+'.jpg', tmp_target_models['SLBRModel'](noise_func.delta_x.data + clean_input)[0][0].detach().cpu().numpy().transpose(1,2,0)*255)
            del tmp_outputs
        
        # make sure adversarial noise inside the correct range
        noise_func.delta_x.data=torch.clamp((torch.clamp(noise_func.delta_x.detach()+clean_input,min=0,max=1)-clean_input),min=-self.budget,max=self.budget).detach()
        input_with_final_noise=noise_func(clean_input)
        if 'SLBRModel' in tmp_target_models:
            assert cv2.imwrite('inpainting/ensemble/intermediate_output/wr_after_final_norm'+str(iter_index)+'.jpg', tmp_target_models['SLBRModel'](input_with_final_noise['SLBRModel'])[0][0].detach().cpu().numpy().transpose(1,2,0)*255)
            
        # return the final adversarial examples
        input_with_final_noise_numpy = input_with_final_noise[list(input_with_final_noise.keys())[0]].clone().detach().cpu().numpy()

        del input,original_outputs,grads, noise_func,attack_optimizer, scheduler, mask, inter_grad, input_with_noise, input_with_final_noise, clean_input
        
        return input_with_final_noise_numpy

    def get_gradient_of_output(self,input):
        tmp_input={}
        
        a= torch.Tensor([[1,1,1],[1,-8,1],[1,1,1]])
        a= torch.Tensor([[0.5,0.5,0.5,0.5,0.5],[0.5,1,1,1,0.5],[0.5,1,-16,1,0.5],[0.5,1,1,1,0.5],[0.5,0.5,0.5,0.5,0.5]])
        a=a.view((5,5)).to(self.device)
        v=torch.zeros((3,3,5,5)).to(self.device)
        for i in range(3):
            v[i,i,:,:]=a

        
        G_x={}
        if isinstance(input,dict):
            for name in input.keys():
                G_x[name]=F.conv2d(input[name],v,padding=int(v.size()[-1]/2),bias=None)
        else:
            G_x=F.conv2d(input,v,padding=int(v.size()[-1]/2),bias=None)
        
        # G=G_x.to(self.device)
        # G=(G/(torch.max(torch.abs(G).view(3,-1),dim=1)[0].view(1,3,1,1).to(self.device)))
        # G=torch.clamp(G,min=-1,max=1) 
        return G_x
    
    def img_gradient_loss(self,original_output,attacked_output,rect,mask):
        mse_fn=torch.nn.MSELoss()

        if 'distance_dependence' in self.opt.algorithm:
            distances={}
            for name in original_output:
                # print(name)
                # exit()
                y_index=torch.range(1,original_output[name].size()[-2])
                x_index=torch.range(1,original_output[name].size()[-1])
                x_index=x_index.view(1,1,1,-1).expand(original_output[name].size()[0],original_output[name].size()[1],original_output[name].size()[2],-1)
                y_index=y_index.view(1,1,-1,1).expand(original_output[name].size()[0],original_output[name].size()[1],-1,original_output[name].size()[3])

                rect_x1=torch.ones(x_index.size())*rect[:,:,0]
                rect_y1=torch.ones(y_index.size())*rect[:,:,2]
                rect_x_size=torch.ones(x_index.size())*rect[:,:,1]

                rect_x2=torch.ones(x_index.size())*(rect[:,:,0]+rect[:,:,1])
                rect_y2=torch.ones(y_index.size())*(rect[:,:,2]+rect[:,:,3])
                rect_y_size=torch.ones(y_index.size())*rect[:,:,3]

                distances[name]=torch.max(1-torch.abs(x_index-rect_x1)/rect_x_size,1-torch.abs(x_index-rect_x2)/rect_x_size)
                distances[name]=torch.max(1-torch.abs(y_index-rect_y1)/rect_y_size,distances[name])
                distances[name]=torch.max(1-torch.abs(y_index-rect_y2)/rect_y_size,distances[name])

                distances[name]=distances[name].to(self.device)*2
            gradient_loss=[]

            original_gradients=self.get_gradient_of_output(original_output)
            attacked_gradients=self.get_gradient_of_output(attacked_output)

            for name in original_gradients:
                original_gradient= original_gradients[name]*distances[name]*mask
                attacked_gradient= attacked_gradients[name]*distances[name] *mask

                gradient_loss.append(mse_fn(original_gradient,attacked_gradient))
            if 'MaxLB' in self.opt.algorithm: 
                gradient_loss=min(gradient_loss)
            else:
                gradient_loss=sum(gradient_loss)
            return gradient_loss
        else:
            gradient_loss=[]
            original_gradients=self.get_gradient_of_output(original_output)
            attacked_gradients=self.get_gradient_of_output(attacked_output)
            
            
            for name in original_gradients:
                print(name,original_gradients[name].size(),mask.size())
                original_gradient=original_gradients[name]*torchvision.transforms.Resize((original_gradients[name].size()[-2],original_gradients[name].size()[-1]))(mask)
                attacked_gradient=attacked_gradients[name]*torchvision.transforms.Resize((original_gradients[name].size()[-2],original_gradients[name].size()[-1]))(mask)

                gradient_loss.append(mse_fn(original_gradient,attacked_gradient))

            gradient_loss=sum(gradient_loss)
            return gradient_loss
            
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--source_model',type=str,required=False,help='the surrogant model')    
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
    opt=parser.parse_args()

    f1=torch.ones((1,3,256,256))
    f2=torch.ones((1,3,256,256))

    
    attacker=Attacker(opt,target_models={})
    loss=attacker.similarity_inverse_loss([f1],[f2])
    # print(loss)
