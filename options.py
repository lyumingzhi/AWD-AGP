import argparse
from email.policy import default
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target_models',action='append',default=[],help='the surrogant model')    
    parser.add_argument('--attackType',action='append',default=[],required=False,help='the attack method')
    parser.add_argument('--algorithm',action='append',default=[],type=str,help='algorithmf for attack')
    parser.add_argument('--lossType',action='append',default=[],help='loss type for attack')
    parser.add_argument('--random_mask',action='store_true',help='whehter use random mask when attack')
    parser.add_argument('--sign',action='store_true',help='whehter use sign of gradient when doing attack')
    parser.add_argument('--InputImg_dir',type=str,required=False,help='the path to input images')
    parser.add_argument('--target_img',type=str,default='',help='target image for target attack')
    parser.add_argument('--target_attack',action='store_true',help='whether to use target attack')
    parser.add_argument('--output_dir',type=str,required=False,help='path to output images')
    parser.add_argument('--attack_specific_region',action='store_true',help='whether to just attack specific region')
    parser.add_argument('--dataset',type=str,default='celeba',help='which dataset the target models are trained on')
    parser.add_argument('--attack_type',action='append',default=[],help='which attack objective is used')
    parser.add_argument('--wo_mask',action='store_true',default=False,help='whether to use raw output of inpainting models')
    parser.add_argument('--iter_num',type=int,default=15,help='num of the iteration to generate adversarial examples')

    parser.add_argument('--RPNRefineMask', type = str, default='', help='wether use RPN Refined Masks')

    parser.add_argument('--get_logo', action='store_true', default=False)
    parser.add_argument('--attach_logo', action='store_true', default=False)

    parser.add_argument('--validation_model', action='append', default=[])

    parser.add_argument('--budget', type=float, default=0.03)

    parser.add_argument('--relative_mask_size', type=float, default=0.33)

    # ADMM
    parser.add_argument('--lr_e',type=float,default=0.02)
    parser.add_argument('--lr_g',type=float,default=0.1)
    parser.add_argument('--rho1',type=float,default=0.005)
    parser.add_argument('--rho2',type=float,default=0.005)
    parser.add_argument('--rho3',type=float,default=0.005)
    parser.add_argument('--rho4',type=float,default=0.0001)
    parser.add_argument('--k',type=int,default=1000,help='num of pixels of the mask')
    parser.add_argument('--lr_min',type=float,default=0.00001,help='min of lr of optimizing mask')
    parser.add_argument('--maxIter_g',type=int,default=1000,help='max iteration of searching for mask for once')
    parser.add_argument('--maxIter_e',type=int,default=15,help='max iteration of searching for epsilon for once')
    parser.add_argument('--rho1_max',type=float,default=20)
    parser.add_argument('--rho2_max',type=float,default=20)
    parser.add_argument('--rho3_max',type=float,default=100)
    parser.add_argument('--rho4_max',type=float,default=0.01)
    parser.add_argument('--lambda1',type=float,default=0.001)
    parser.add_argument('--rho_increase_factor',type=float,default=1.01)
    parser.add_argument('--lr_decay_step',type=int,default=50)
    parser.add_argument('--lr_decay_factor',type=float,default=0.9)
    parser.add_argument('--max_search_times',type=int,default=15)
    parser.add_argument('--maxIter_mm',type=int,default=15)
    parser.add_argument('--rho_increase_step',type=int,default=1)
    
    opt=parser.parse_args()
    return opt