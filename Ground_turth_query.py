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
from inpainting.ensemble.utils import rbf_interpolation, ND_interp, get_superpixel, get_semantic_loss

from scipy.optimize import differential_evolution


class Ground_turth_query():
    def __init__(self, opt):
        self.opt=opt
        # opt=get_args()

        RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet, _, _, _=load_models(opt)
 
        all_models={'Crfillnet':Crfillnet,'Gennet':Gennet,'FcFnet':FcFnet,'Matnet':Matnet}
 
        self.models={}
        self.validation_model={}
        for name in all_models:
            if  name not in opt.validation_model:
                self.models[name]=all_models[name]
            if name in opt.validation_model:
                self.validation_model[name]=all_models[name] 

        self.attacker=Attacker(opt,self.models,validation_model=self.validation_model,iters=opt.iter_num, budget=opt.budget)

    def update_superpixel_map(self, superpixel_map):
        if 'semantic_attack' in self.opt.algorithm:
            # self.superpixel_map, self.superpixel_map_vis=get_superpixel(img)
            self.superpixel_map = superpixel_map


    def __call__(self, img, mask, rect=None, original_FeatList=None):
        
        # original_output={}
        # for name in self.models.keys():
        #     original_output[name]=transforms.Resize(self.models[name](img,mask).size()[-2:])(img) ##### can be activated for learn the output to attack inpainting
        if False:
            if self.opt.target_attack:
                if 'unchange_attack' in self.opt.attack_type:
                    target_patch=img[:,:,rect[0,0,0]:rect[0,0,1]+rect[0,0,0],rect[0,0,2]:rect[0,0,2]+rect[0,0,3]]
                    input_with_final_noise,adv_noise,attacked_output=self.attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)
                elif self.opt.attack_specific_region:
                    print('start to attack')
                    input_with_final_noise,adv_noise,attacked_output=self.attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)    
                else:
                    input_with_final_noise,adv_noise,attacked_output,output_mask=self.attacker.attack(img,original_output,mask,target_patch=target_patch,rect=rect,original_FeatList=original_FeatList)
            else:    
                # input_with_final_noise,adv_noise,attacked_output,output_mask=attacker.attack(img,original_output,mask,rect=rect,original_FeatList=original_FeatList)
                # tmp_input_with_final_noise,tmp_adv_noise,tmp_attacked_output,tmp_output_mask,tmp_validation_output,tmp_original_output_for_validation,validation_score=attacker.attack(img,original_output,mask,rect=rect,original_FeatList=original_FeatList)
                tmp_attacked_output,attacked_score=self.attacker.evaluate(img,None,original_output,mask,rect=rect,original_FeatList=original_FeatList)
        tmp_attacked_output=None
        if 'semantic_attack' in self.opt.algorithm:
            semantic_loss=get_semantic_loss(self.superpixel_map,mask)
            attacked_score=semantic_loss #overwrite the output of attack

        # mask_score_map[mask_index[0],mask_index[1]]=attacked_score
        # activated_mask[mask_index[0],mask_index[1]]=1
        # print('validation score',attacked_score,best_validation_score)
        # print('mask score map',torch.sum(mask_score_map))
        # exit()
        # if attacked_score<=best_validation_score:
        #     best_validation_score=attacked_score
        #     # input_with_final_noise=tmp_input_with_final_noise
        #     # adv_noise=tmp_adv_noise
        #     attacked_output=tmp_attacked_output
        #     output_mask=mask

        return attacked_score