import argparse

# from nbformat import write
import torch
import cv2
import json
import torchvision
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import pandas as pd 
import numpy as np
import os
from inpainting.ensemble.official_dataset import AttackDataset, TransferAttackDataset
from torch.utils.data import DataLoader


def calc_Content_Loss(features, targets, weights=None):
    mse_criterion=torch.nn.MSELoss(reduction='mean')
    if weights is None:
        weights = [1/len(features)] * len(features)
    
    content_loss = 0
    if isinstance(features,dict):
        for name in features.keys():
            for f, t, w in zip(features[name], targets[name], weights):
                content_loss += mse_criterion(f, t) * w
    else:
        for f, t, w in zip(features , targets , weights):
            content_loss += mse_criterion(f, t) * w       
    return content_loss
def calc_Gram_Loss(features, targets, weights=None):
    mse_criterion=torch.nn.MSELoss(reduction='mean')
    if weights is None:
        weights = [1/len(features)] * len(features)
        
    gram_loss = 0
    if isinstance(features,dict):
        for name in features.keys():
            for f, t, w in zip(features[name], targets[name], weights):
               gram_loss += mse_criterion(gram(f), gram(t)) * w
    else:
        for f, t, w in zip(features, targets, weights):
            gram_loss += mse_criterion(gram(f), gram(t)) * w
    return gram_loss

def gram(x):
    b ,c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1,2))
    return g.div(h*w)
def extract_features( model, x, layers):
    All_features={}
    
    if isinstance(x,dict):
        for name in x.keys():
            features = list()
            single_x=x[name]
            # print(single_x.size())
            # exit()
            for index, layer in enumerate(model):
                single_x= layer(single_x)
                if index in layers:
                    features.append(single_x)
            All_features[name]=features
    else:
        features = list()
        single_x=x.cuda()
        
        for index, layer in enumerate(model):
            single_x= layer(single_x)
            if index in layers:
                features.append(single_x)
        All_features=features
    return All_features
def get_perceputal_loss(original_output,tmp_output,noise_func=None,loss_vgg=None,vgg_content_layers=[15],vgg_style_layers=[3,8,15,22],target_img=None):
 
    if target_img!=None:
        perceptual_loss=0
        target_output_content_features=extract_features(loss_vgg,target_img.unsqueeze(0),vgg_content_layers)
        attacked_output_content_features=extract_features(loss_vgg,tmp_output,vgg_content_layers)

        target_output_style_features=extract_features(loss_vgg,target_img.unsqueeze(0),vgg_style_layers)
        attacked_output_style_features=extract_features(loss_vgg,tmp_output,vgg_style_layers)

        perceptual_loss+=1.0*calc_Content_Loss(target_output_content_features,attacked_output_content_features)
        perceptual_loss+=3.0*calc_Gram_Loss(target_output_style_features,attacked_output_style_features)
        
    
        return perceptual_loss.detach().cpu().item()
    else:
        perceptual_loss=0
        original_output_content_features=extract_features(loss_vgg,original_output,vgg_content_layers)
        attacked_output_content_features=extract_features(loss_vgg,tmp_output,vgg_content_layers)

        original_output_style_features=extract_features(loss_vgg,original_output,vgg_style_layers)
        attacked_output_style_features=extract_features(loss_vgg,tmp_output,vgg_style_layers)

        perceptual_loss+=1.0*calc_Content_Loss(original_output_content_features,attacked_output_content_features)
        perceptual_loss+=3.0*calc_Gram_Loss(original_output_style_features,attacked_output_style_features)
    
   
        return perceptual_loss.detach().cpu().item()
def PSNR(original,attacked,target_img=None):
    # print('size',(original-compressed).shape)
    # difference=(original-attacked).numpy()*255
    
    if target_img!=None:
        mse=torch.mean((target_img*255-attacked*255 )**2)
    else:
        mse=torch.mean((original*255-attacked*255 )**2)
    if (mse==0):
        return 100
    max_pixel=255
    
    psnr=20*torch.log10(max_pixel/torch.sqrt(mse))

    return psnr.item()
def compute_ssim3(original, attacked,target_img=None):
    # original=torch.from_numpy(original)
    # attacked=torch.from_numpy(attacked)
    if target_img!=None:
        target_img_gray=torchvision.transforms.functional.rgb_to_grayscale(target_img)
        attacked_gray=torchvision.transforms.functional.rgb_to_grayscale(attacked)

        target_img=target_img_gray.clone().numpy().transpose(1,2,0)*255
        attacked=attacked_gray.clone().numpy().transpose(1,2,0)*255
        # print(original.shape)
        ssim_value=ssim(target_img[...,0],attacked[...,0],multichannel=True,data_range=255).item()

        return ssim_value
    else:
        original_gray=torchvision.transforms.functional.rgb_to_grayscale(original)
        attacked_gray=torchvision.transforms.functional.rgb_to_grayscale(attacked)

        original=original_gray.clone().numpy().transpose(1,2,0)*255
        attacked=attacked_gray.clone().numpy().transpose(1,2,0)*255
        ssim_value=ssim(original[...,0],attacked[...,0],multichannel=True,data_range=255).item()

        return ssim_value
    
def get_mse(attack_patch,original_patch):
    mse_fn=torch.nn.MSELoss() 

    mse_loss=mse_fn(attack_patch,original_patch)
    return mse_loss.item()

def get_edge_mse(original_output,attacked_output):
    
    if original_output.dim()==3:
        original_output = original_output.unsqueeze(0)
    if attacked_output.dim()==3:
        attacked_output = attacked_output.unsqueeze(0)
    original_grad = get_img_edge(original_output)
    attacked_grad = get_img_edge(attacked_output)
    masked_original_grad = original_grad
    masked_attacked_grad = attacked_grad
    mse_fn=torch.nn.MSELoss()

    mse_loss=mse_fn(masked_original_grad,masked_attacked_grad)
    return mse_loss.item()
    
import kornia as K
def get_img_edge(img):

    images_gray=torchvision.transforms.functional.rgb_to_grayscale(img)
    edges=K.filters.canny(images_gray,sigma=(2,2),low_threshold=0.05,high_threshold=0.1,kernel_size=(21,21))[1]
    # assert cv2.imwrite('inpainting/ensemble/edge_example.jpg', edges[0].detach().cpu().numpy().transpose(1,2,0)*255)
    # # print(torch.max(img))
    # assert cv2.imwrite('inpainting/ensemble/edge_input_example.jpg', img[0].detach().cpu().numpy().transpose(1,2,0)*255)
    return edges
    
def preprocess(image):
    image=transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((256,256)),
                            ])(image)
    return image

opt = None
def save_detail_records(**records):
    detail_records = {key: value for key, value in records.items()}
    detail_df =  {'model':[]}
    for name in detail_records[list(detail_records.keys())[0]]:
        for metric, records in detail_records.items():
            if metric not in detail_df:
                detail_df[metric]=records[name]
                
            else:
                detail_df[metric].extend(records[name])
        detail_df['model'].extend([name for i in detail_records[list(detail_records.keys())[0]][name]])  
    # print(detail_df)
    writer = pd.ExcelWriter(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'detail_'+name+'_maskded_only_result_with_input_wo_mask.xlsx'), engine='xlsxwriter')

    f=open(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'detail_'+name+'_maskded_only_result_with_input_wo_mask.json'),'w')
    json.dump(detail_df,f,indent=4)
    # with open(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,name+'_result.json'),'w') as f:
    #     json.dump(df,f,indent=4)
    detail_df=pd.DataFrame(detail_df)
    detail_df.to_excel(writer,sheet_name='Sheet1',index=False)
    writer.save()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target_models',action='append',required=False,help='the surrogant model')
    parser.add_argument('--output_dir',type=str,required=True,help='experiment name (must exist)')
    parser.add_argument('--source_dir',type=str,help='required for retriving the original mask images')
    parser.add_argument('--target_attack',action='store_true')
    parser.add_argument('--target_img',type=str,help='target image when doing target attack')
    parser.add_argument('--algorithm',action='append',default=[])
    parser.add_argument('--InputImg_dir',type=str,required=False,help='the path to input images')
    parser.add_argument('--RPNRefineMask', type=str, default=None, help='whether to use mask from RPN')
    parser.add_argument('--experiment_dir',type=str,default='',help='experiment name of the adversarial generation experiment')
    parser.add_argument('--get_logo', action = 'store_true', default=False, help='whether to get a example logo')
    parser.add_argument('--attach_logo', action = 'store_true', default=False, help='whether to attach logo to the dataset image')
    parser.add_argument('--watermark_alpha', type = int, default=0.5, help ='the alpha to embed watermark to the image')
    parser.add_argument('--lossType',action='append',default=['perceptual_loss'],help='loss type for attack')

    global opt
    opt=parser.parse_args()
    


    loss_vgg=torchvision.models.__dict__['vgg16'](pretrained=True).features.to('cuda')

    with open('inpainting/ensemble/experiment_result/'+opt.output_dir+'/rect.json') as f:
        rect=json.load(f)

    SSIM_final_result={}
    PSNR_final_result={}
    perceptual_result={}
    mse_final_result={}
    edge_mse_final_result={}
    mask_mse_result = {}
    # mask_from_wrm_mse_result = {}

    SSIM_final_result_att_output_ori_input={}
    PSNR_final_result_att_output_ori_input={}
    perceptual_result_att_output_ori_input={}
    mse_final_result_att_output_ori_input={}
    edge_mse_final_result_ori_output_ori_input={}

    SSIM_final_result_att_output_ori_input_wo_logo={}
    PSNR_final_result_att_output_ori_input_wo_logo={}
    perceptual_result_att_output_ori_input_wo_logo={}
    mse_final_result_att_output_ori_input_wo_logo={}
    edge_mse_final_result_ori_output_ori_input_wo_logo={}

    SSIM_final_result_att_final_output_ori_final_output={}
    PSNR_final_result_att_final_output_ori_final_output={}
    perceptual_final_result_att_final_output_ori_final_output={}
    mse_final_result_att_final_output_ori_final_output={}
    edge_mse_final_result_att_final_output_ori_final_output={}
    
    if 'random_mask' in opt.algorithm:
        random_mask_SSIM_final_result={}
        random_mask_PSNR_final_result={}
        random_mask_perceptual_result={}
        random_mask_mse_final_result={}

    for name in opt.target_models:
        SSIM_final_result[name]=[]
        PSNR_final_result[name]=[]
        perceptual_result[name]=[]
        mse_final_result[name]=[]
        edge_mse_final_result[name]=[]
        mask_mse_result[name] = []
        # mask_from_wrm_mse_result[name] = []

        SSIM_final_result_att_output_ori_input[name]=[]
        PSNR_final_result_att_output_ori_input[name]=[]
        perceptual_result_att_output_ori_input[name]=[]
        mse_final_result_att_output_ori_input[name]=[]
        edge_mse_final_result_ori_output_ori_input[name]=[]

        SSIM_final_result_att_output_ori_input_wo_logo[name]=[]
        PSNR_final_result_att_output_ori_input_wo_logo[name]=[]
        perceptual_result_att_output_ori_input_wo_logo[name]=[]
        mse_final_result_att_output_ori_input_wo_logo[name]=[]
        edge_mse_final_result_ori_output_ori_input_wo_logo[name]=[]

        SSIM_final_result_att_final_output_ori_final_output[name] = []
        PSNR_final_result_att_final_output_ori_final_output[name] = []
        perceptual_final_result_att_final_output_ori_final_output[name] = []
        mse_final_result_att_final_output_ori_final_output[name] = []
        edge_mse_final_result_att_final_output_ori_final_output[name] = []

        if 'random_mask' in opt.algorithm:
            random_mask_SSIM_final_result[name]=[]
            random_mask_PSNR_final_result[name]=[]
            random_mask_perceptual_result[name]=[]
            random_mask_mse_final_result[name]=[]
    
    mask_paths=[]
    
    print(os.path.join('inpainting/ensemble/experiment_result/',opt.source_dir,'Gennet','attacked_mask/'))
    
    mask_index=0
    for root,dirs,files in os.walk(os.path.join('inpainting/ensemble/experiment_result/',opt.source_dir,'Matnet','attacked_mask/')):
        for file in files:
            mask_paths.append(os.path.join(root,'mask__'+str(mask_index)+'.png'))
            mask_index+=1
    
    random_mask_paths=[]
    original_inputs_paths=[]
    if 'random_mask' in opt.algorithm:
        mask_index=0
        for root,dirs,files in os.walk(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'random_mask/')):
            for file in files:
                random_mask_paths.append(os.path.join(root,'random_mask_'+str(mask_index)+'.png'))
                mask_index+=1
    # for root,dirs,files in os.walk(opt.InputImg_dir):
    #     for file in files:
            
    #         original_inputs_paths.append(os.path.join(root,file))
    
    # for i in range(len(rect['FcFnet'])):
    
    # for i in range(len((rect[list(rect.keys())[0]]))):
    
    for name in opt.target_models:
        print('name to test', name)
        # if i>134:
        #     continue
        name_to_test = name
        opt.experiment_dir = os.path.join('inpainting/ensemble/experiment_result', opt.source_dir)
        dataset=TransferAttackDataset(opt,opt.InputImg_dir,opt.experiment_dir,name_to_test, if_sort=True)
        dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
        dataloader = iter(dataloader)
        # print('dataset len', len(dataset), len(mask_paths))
        # exit()
        original_output={}
        attacked_output={}
        for i,mask_path in enumerate(mask_paths):
            if i >= 200:
                break
            
            # attacked_img=cv2.imread('inpainting/ensemble/experiment_result/'+opt.output_dir+'/result/attacked_'+name+'_'+str(i)+'.png')
            # original_img=cv2.imread('inpainting/ensemble/experiment_result/'+opt.output_dir+'/result/original_'+name+'_'+str(i)+'.png')

            print('current file', os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/attacked_'+name+'_'+str(i)+'.png'))
            attacked_img=cv2.imread(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/attacked_'+name+'_'+str(i)+'.png')).astype(float)/255
            
            original_img=cv2.imread(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/original_'+name+'_'+str(i)+'.png')).astype(float)/255

            # original_mask = cv2.imread(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/original_pred_mask_'+name+'_'+str(i)+'.png')).astype(float)/255

            # attacked_mask = cv2.imread(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/attacked_pred_mask_'+name+'_'+str(i)+'.png')).astype(float)/255

            original_pred_logo_mask = cv2.imread(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/original_pred_mask_'+name+'_'+str(i)+'.png'))
            original_pred_logo_mask = cv2.cvtColor(original_pred_logo_mask, cv2.COLOR_BGR2GRAY).astype(float)/255
            attacked_pred_logo_mask = cv2.imread(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/attacked_pred_mask_'+name+'_'+str(i)+'.png'))
            attacked_pred_logo_mask = cv2.cvtColor(attacked_pred_logo_mask, cv2.COLOR_BGR2GRAY).astype(float)/255

            original_final_img = cv2.imread(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/final_original_'+name+'_'+str(i)+'.png')).astype(float)/255
            attacked_final_img = cv2.imread(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,'result/final_attacked_'+name+'_'+str(i)+'.png')).astype(float)/255

            if attacked_img is None or original_img is None and original_final_img is not None and attacked_final_img is not None:
                break

            # if 'random_mask'  in opt.algorithm:
            # original_input=cv2.imread(original_inputs_paths[i])
            original_input, _, _, rect, _, _, original_input_wo_logo= next(dataloader) #rect size = [f0, size0, f1, size1]

            rect = rect.numpy().astype(int)
            original_input = (original_input[0].cpu().numpy().transpose(1,2,0))
            original_input_wo_logo = (original_input_wo_logo[0].cpu().numpy().transpose(1,2,0))
            
            
            # assert cv2.imwrite('inpainting/ensemble/original_input_in_eval.jpg', original_input)
            # assert cv2.imwrite('inpainting/ensemble/attacked_img_in_eval.jpg', attacked_img)
            # assert cv2.imwrite('inpainting/ensemble/original_img_in_eval.jpg', original_img)
            # print`(rect.keys())
            # if i == 2:
            #     exit()  
            # rect[name]=rect['FcFnet']
            if opt.target_attack:
                print(opt.target_img,'is target attack')
                target_img=cv2.imread(opt.target_img)
                
                print(len(rect[name]),i,name,rect.keys())
                # exit(0)
                
                target_img=cv2.resize(target_img,(rect[name][i][1],rect[name][i][3]))
                target_img=preprocess(target_img)/255

            # attacked_patch=attacked_img[rect[name][i][0]:rect[name][i][0]+rect[name][i][1],rect[name][i][2]:rect[name][i][2]+rect[name][i][3],:]
            # original_patch=original_img[rect[name][i][0]:rect[name][i][0]+rect[name][i][1],rect[name][i][2]:rect[name][i][2]+rect[name][i][3],:]
            
            # print('mask path',mask_path)
            # mask=transforms.ToTensor()(cv2.imread(mask_path))
            # mask=((torch.sum(mask,dim=0)/3)>0.9)*1.0
            mask=cv2.resize(cv2.imread(mask_path),(attacked_img.shape[0],attacked_img.shape[0])).astype('float')/255
            # mask=(np.sum(mask,axis=-1)/255/3)
            mask=(mask>0.9)*1.0
            
            if 'random_mask' in opt.algorithm:
                random_mask=cv2.resize(cv2.imread(random_mask_paths[i]),(attacked_img.shape[0],attacked_img.shape[0])).astype('float')/255
                random_mask=(random_mask>0.9)*1.0

            
            # attacked_patch=attacked_img*mask
            # original_patch=original_img*mask
            enlarge_ratio = (attacked_img.shape[0]/original_input.shape[0], attacked_img.shape[1]/original_input.shape[1])


            rect[...,:2] = (rect[..., :2] * enlarge_ratio[0]).astype(int)
            rect[...,2:] = (rect[..., 2:] * enlarge_ratio[1]).astype(int)
            attacked_patch = attacked_img[rect.item(0):rect.item(0)+rect.item(1), rect.item(2):rect.item(2)+rect.item(3),:]
            original_patch = original_img[rect.item(0):rect.item(0)+rect.item(1), rect.item(2):rect.item(2)+rect.item(3),:]

            attacked_final_patch = attacked_final_img
            original_final_patch = original_final_img

            
            attacked_patch=preprocess(attacked_patch).float()
            original_patch=preprocess(original_patch).float()

            attacked_final_patch=preprocess(attacked_final_patch).float()
            original_final_patch=preprocess(original_final_patch).float()

            # assert cv2.imwrite('inpainting/ensemble/original_patch.jpg', original_patch.detach().cpu().numpy().transpose(1,2,0)*255)
            assert torch.max(attacked_patch)<=1.0 and torch.max(original_patch)<=1.0
            # attacked_patch = original_patch


            attacked_mask_patch = preprocess(attacked_pred_logo_mask).float()
            original_mask_patch = preprocess(original_pred_logo_mask).float()
            # assert torch.max(attacked_mask_patch) <= 1.0 and torch.max(original_mask_patch) <= 1.0

            attacked_mask_patch = (attacked_mask_patch>0.5)*1.0
            original_mask_patch = (original_mask_patch>0.5)*1.0
            
            
            original_input=cv2.resize(original_input,(attacked_img.shape[0],attacked_img.shape[0]))
            original_input_wo_logo = cv2.resize(original_input_wo_logo,(attacked_img.shape[0],attacked_img.shape[0]))
            
           
            # assert cv2.imwrite('inpainting/ensemble/masked_original_inputs.jpg', original_input*mask*255)
            original_inputs_patch = original_input[rect.item(0):rect.item(0)+rect.item(1), rect.item(2):rect.item(2)+rect.item(3),:]
            print(rect.item(0),rect.item(0)+rect.item(1), rect.item(2),rect.item(2)+rect.item(3),)
            original_inputs_patch=preprocess(original_inputs_patch).float()

            original_inputs_wo_logo_patch = original_input_wo_logo[rect.item(0):rect.item(0)+rect.item(1), rect.item(2):rect.item(2)+rect.item(3),:]
            original_inputs_wo_logo_patch=preprocess(original_inputs_wo_logo_patch).float()

            # assert cv2.imwrite('inpainting/ensemble/original_inputs_patch.jpg', original_inputs_patch.detach().cpu().numpy().transpose(1,2,0)*255)

            assert torch.max(original_inputs_patch) <= 1.0
            assert torch.max(original_inputs_wo_logo_patch) <= 1.0
            
            # attacked_patch = original_inputs_patch

            if opt.target_attack:
                psnr=PSNR(attacked_patch,original_patch,target_img=target_img)    
            else:
                if 'random_mask' in opt.algorithm:
                    psnr=PSNR(attacked_patch,preprocess(original_input*mask).float()/255) 

                    random_mask_psnr=PSNR(preprocess(original_img*random_mask).float()/255,preprocess(original_input*random_mask).float()/255) 

                else:
                    psnr=PSNR(attacked_patch,original_patch)
                    psnr_att_output_ori_input=PSNR(attacked_patch,original_inputs_patch)
                    print('psnr_att_output_ori_input', psnr_att_output_ori_input)
                    psnr_att_output_ori_input_wo_logo=PSNR(attacked_patch,original_inputs_wo_logo_patch)
                    psnr_att_final_output_ori_final_output = PSNR(attacked_final_patch,original_final_patch)

                    # assert cv2.imwrite('inpainting/ensemble/example_attach_patch.jpg', attacked_patch.detach().cpu().numpy().transpose(1,2,0) * 255)
                    # assert cv2.imwrite('inpainting/ensemble/example_original_patch.jpg', original_patch.detach().cpu().numpy().transpose(1,2,0) * 255)
                    # assert cv2.imwrite('inpainting/ensemble/example_original_inputs_patch.jpg', original_inputs_patch.detach().cpu().numpy().transpose(1,2,0) * 255)
                    # assert cv2.imwrite('inpainting/ensemble/example_original_inputs_wo_logo_patch.jpg', original_inputs_wo_logo_patch.detach().cpu().numpy().transpose(1,2,0) * 255)
                    # assert cv2.imwrite('inpainting/ensemble/example_attacked_final_patch.jpg', attacked_final_patch.detach().cpu().numpy().transpose(1,2,0) * 255)
                    # assert cv2.imwrite('inpainting/ensemble/example_original_final_patch.jpg', original_final_patch.detach().cpu().numpy().transpose(1,2,0) * 255)
                    # exit()

            if opt.target_attack:
                ssim=compute_ssim3(attacked_patch,original_patch,target_img=target_img)
            else:
                if 'random_mask' in opt.algorithm:
                    ssim=compute_ssim3(attacked_patch,preprocess(original_input*mask).float()/255) 

                    random_mask_ssim=compute_ssim3(preprocess(original_img*random_mask).float()/255,preprocess(original_input*random_mask).float()/255) 
                else:
                    ssim=compute_ssim3(attacked_patch,original_patch)
                    ssim_att_output_ori_input=compute_ssim3(attacked_patch,original_inputs_patch)
                    ssim_att_output_ori_input_wo_logo=compute_ssim3(attacked_patch,original_inputs_wo_logo_patch)
                    ssim_att_final_output_ori_final_output = compute_ssim3(attacked_final_patch,original_final_patch)

            if opt.target_attack:
                perceptual_loss=get_perceputal_loss(attacked_patch.unsqueeze(0).cuda(),original_patch.unsqueeze(0).cuda(),loss_vgg=loss_vgg,target_img=target_img)
            else:      
                if 'random_mask' in opt.algorithm:
                    perceptual_loss=get_perceputal_loss(attacked_patch.unsqueeze(0).cuda(),(preprocess(original_input*mask).float()/255).unsqueeze(0).cuda(),loss_vgg=loss_vgg) 

                    random_mask_perceptual_loss=get_perceputal_loss((preprocess(original_img*random_mask).float()/255).unsqueeze(0).cuda(),(preprocess(original_input*random_mask).float()/255).unsqueeze(0).cuda(),loss_vgg=loss_vgg) 
                else:
                    perceptual_loss=get_perceputal_loss(attacked_patch.unsqueeze(0).cuda(),original_patch.unsqueeze(0).cuda(),loss_vgg=loss_vgg)

                    perceptual_loss_att_output_ori_input=get_perceputal_loss(attacked_patch.unsqueeze(0).cuda(),original_inputs_patch.unsqueeze(0).cuda(),loss_vgg=loss_vgg)
                    # cv2.imwrite('inpainting/ensemble/attacked_patch_in_evaluate_wr.jpg', attacked_patch.detach().cpu().numpy().transpose(1,2,0)*255)
                    # cv2.imwrite('inpainting/ensemble/original_input_patch_in_evaluate_wr.jpg', original_inputs_patch.detach().cpu().numpy().transpose(1,2,0)*255)
                    # exit()

                    perceptual_loss_att_output_ori_input_wo_logo=get_perceputal_loss(attacked_patch.unsqueeze(0).cuda(),original_inputs_wo_logo_patch.unsqueeze(0).cuda(),loss_vgg=loss_vgg)

                    perceptual_loss_att_final_output_ori_final_output=get_perceputal_loss(attacked_final_patch.unsqueeze(0).cuda(),original_final_patch.unsqueeze(0).cuda(),loss_vgg=loss_vgg)
                    

            if opt.target_attack:
                mse_loss=get_mse(attacked_patch.unsqueeze(0).cuda(),original_patch.unsqueeze(0).cuda(),target_img=target_img)
            else:
                if 'random_mask' in opt.algorithm:
                    mse_loss=get_mse(attacked_patch.unsqueeze(0).cuda(),original_patch.unsqueeze(0).cuda())
                    random_mask_mse_loss=get_mse((preprocess(original_img*random_mask).float()/255).unsqueeze(0).cuda(),(preprocess(original_input*random_mask).float()/255).unsqueeze(0).cuda())
                else:
                    mse_loss=get_mse(attacked_patch.unsqueeze(0).cuda(),original_patch.unsqueeze(0).cuda())
                    mse_loss_att_output_ori_input=get_mse(attacked_patch.unsqueeze(0).cuda(),original_inputs_patch.unsqueeze(0).cuda())
                    mse_loss_att_output_ori_input_wo_logo = get_mse(attacked_patch.unsqueeze(0).cuda(),original_inputs_wo_logo_patch.unsqueeze(0).cuda())

                    mse_loss_att_mask_ori_mask = get_mse(attacked_mask_patch.unsqueeze(0).cuda(),original_mask_patch.unsqueeze(0).cuda())

                    mse_loss_att_final_output_ori_final_output = get_mse(attacked_final_patch.unsqueeze(0).cuda(),original_final_patch.unsqueeze(0).cuda())


            if opt.target_attack:
                edge_mse_loss=get_edge_mse(target_img,original_patch)
            else:
                if 'random_mask' in opt.algorithm:
                    edge_mse_loss=get_edge_mse(attacked_patch,original_patch)
                    random_edge_mask_mse_loss=get_edge_mse((preprocess(original_img*random_mask).float()/255),(preprocess(original_input*random_mask).float()/255))
                else:
                    edge_mse_loss=get_edge_mse(attacked_patch,original_patch)
                    edge_mse_loss_att_output_ori_input=get_edge_mse(attacked_patch,original_inputs_patch)
                    edge_mse_loss_att_output_ori_input_wo_logo=get_edge_mse(attacked_patch,original_inputs_wo_logo_patch)
                    edge_mse_loss_att_final_output_ori_final_output=get_edge_mse(attacked_final_patch,original_final_patch)

                    print('edge_mse_loss_att_output_ori_input_wo_logo', edge_mse_loss_att_output_ori_input_wo_logo)
            # pred_mask_from_wrm_mse = get_mse(original_pred_logo_mask.unsqueeze(0).cuda(),attacked_pred_logo_mask.unsqueeze(0).cuda())

            
            

            SSIM_final_result[name].append(ssim)
            PSNR_final_result[name].append(psnr)
            perceptual_result[name].append(perceptual_loss)
            mse_final_result[name].append(mse_loss)
            edge_mse_final_result[name].append(edge_mse_loss)
            mask_mse_result[name].append(mse_loss_att_mask_ori_mask)
            # mask_from_wrm_mse_result[name].append(pred_mask_from_wrm_mse)

            SSIM_final_result_att_output_ori_input[name].append(ssim_att_output_ori_input)
            PSNR_final_result_att_output_ori_input[name].append(psnr_att_output_ori_input)
            perceptual_result_att_output_ori_input[name].append(perceptual_loss_att_output_ori_input)
            mse_final_result_att_output_ori_input[name].append(mse_loss_att_output_ori_input)
            edge_mse_final_result_ori_output_ori_input[name].append(edge_mse_loss_att_output_ori_input)

            SSIM_final_result_att_output_ori_input_wo_logo[name].append(ssim_att_output_ori_input_wo_logo)
            PSNR_final_result_att_output_ori_input_wo_logo[name].append(psnr_att_output_ori_input_wo_logo)
            perceptual_result_att_output_ori_input_wo_logo[name].append(perceptual_loss_att_output_ori_input_wo_logo)
            mse_final_result_att_output_ori_input_wo_logo[name].append(mse_loss_att_output_ori_input_wo_logo)
            edge_mse_final_result_ori_output_ori_input_wo_logo[name].append(edge_mse_loss_att_output_ori_input_wo_logo)

            SSIM_final_result_att_final_output_ori_final_output[name].append(ssim_att_final_output_ori_final_output)
            PSNR_final_result_att_final_output_ori_final_output[name].append(psnr_att_final_output_ori_final_output)
            perceptual_final_result_att_final_output_ori_final_output[name].append(perceptual_loss_att_final_output_ori_final_output)
            mse_final_result_att_final_output_ori_final_output[name].append(mse_loss_att_final_output_ori_final_output)
            edge_mse_final_result_att_final_output_ori_final_output[name].append(edge_mse_loss_att_final_output_ori_final_output)


            if 'random_mask' in opt.algorithm:
                random_mask_SSIM_final_result[name].append(random_mask_ssim)
                random_mask_PSNR_final_result[name].append(random_mask_psnr)
                random_mask_perceptual_result[name].append(random_mask_perceptual_loss)
                random_mask_mse_final_result[name].append(random_mask_mse_loss)
    df={'models':[],
    'SSIM_final_result':[],
    'PSNR_final_result':[],
    'perceptual_result':[],
    'mse_final_result':[],
    'edge_mse_final_result':[],
    'mask_mse_result':[], 
    # 'mask_from_wrm_mse':[], 
    'SSIM_final_result_att_output_ori_input':[],
    'PSNR_final_result_att_output_ori_input':[],
    'perceptual_result_att_output_ori_input':[],
    'mse_final_result_att_output_ori_input':[],
    'edge_mse_final_result_ori_output_ori_input':[],
    
    'SSIM_final_result_att_output_ori_input_wo_logo':[],
    'PSNR_final_result_att_output_ori_input_wo_logo':[],
    'perceptual_result_att_output_ori_input_wo_logo':[],
    'mse_final_result_att_output_ori_input_wo_logo':[],
    'edge_mse_final_result_ori_output_ori_input_wo_logo':[],

    'SSIM_final_result_att_final_output_ori_final_output':[],
    'PSNR_final_result_att_final_output_ori_final_output':[],
    'perceptual_final_result_att_final_output_ori_final_output':[],
    'mse_final_result_att_final_output_ori_final_output':[],
    'edge_mse_final_result_att_final_output_ori_final_output':[],
    }
    if 'random_mask' in opt.algorithm:
        df={'models':[],'SSIM_final_result':[],'PSNR_final_result':[],'perceptual_result':[],'random_mask_SSIM_final_result':[],'random_mask_PSNR_final_result':[],'random_mask_perceptual_result':[]}
    writer = pd.ExcelWriter(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,name+'_maskded_only_result_for_wr.xlsx'), engine='xlsxwriter')

    save_detail_records(
        SSIM_final_result=SSIM_final_result,
        PSNR_final_result=PSNR_final_result,
        perceptual_result=perceptual_result,
        mse_final_result=mse_final_result,
        edge_mse_final_result=edge_mse_final_result,
        mask_mse_result=mask_mse_result,
        SSIM_final_result_att_output_ori_input=SSIM_final_result_att_output_ori_input,
        PSNR_final_result_att_output_ori_input=PSNR_final_result_att_output_ori_input,
        perceptual_result_att_output_ori_input=perceptual_result_att_output_ori_input,
        mse_final_result_att_output_ori_input=mse_final_result_att_output_ori_input,
        edge_mse_final_result_ori_output_ori_input=edge_mse_final_result_ori_output_ori_input,

        SSIM_final_result_att_output_ori_input_wo_logo=SSIM_final_result_att_output_ori_input_wo_logo,
        PSNR_final_result_att_output_ori_input_wo_logo=PSNR_final_result_att_output_ori_input_wo_logo,
        perceptual_result_att_output_ori_input_wo_logo=perceptual_result_att_output_ori_input_wo_logo,
        mse_final_result_att_output_ori_input_wo_logo=mse_final_result_att_output_ori_input_wo_logo,
        edge_mse_final_result_ori_output_ori_input_wo_logo=edge_mse_final_result_ori_output_ori_input_wo_logo,
        SSIM_final_result_att_final_output_ori_final_output=SSIM_final_result_att_final_output_ori_final_output,
        PSNR_final_result_att_final_output_ori_final_output=PSNR_final_result_att_final_output_ori_final_output,
        perceptual_final_result_att_final_output_ori_final_output=perceptual_final_result_att_final_output_ori_final_output,
        mse_final_result_att_final_output_ori_final_output=mse_final_result_att_final_output_ori_final_output,
        edge_mse_final_result_att_final_output_ori_final_output=edge_mse_final_result_att_final_output_ori_final_output,
    )

    for name in opt.target_models:
        SSIM_final_result[name]=sum(SSIM_final_result[name])/len(SSIM_final_result[name])
        PSNR_final_result[name]=sum(PSNR_final_result[name])/len(PSNR_final_result[name])
        perceptual_result[name]=sum(perceptual_result[name])/len(perceptual_result[name])
        mse_final_result[name]=sum(mse_final_result[name])/len(mse_final_result[name])
        edge_mse_final_result[name]=sum(edge_mse_final_result[name])/len(edge_mse_final_result[name])
        mask_mse_result[name]=sum(mask_mse_result[name])/len(mask_mse_result[name])
        # mask_from_wrm_mse_result[name]=sum(mask_from_wrm_mse_result[name]/len(mask_from_wrm_mse_result[name]))

        SSIM_final_result_att_output_ori_input[name]=sum(SSIM_final_result_att_output_ori_input[name])/len(SSIM_final_result_att_output_ori_input[name])
        PSNR_final_result_att_output_ori_input[name]=sum(PSNR_final_result_att_output_ori_input[name])/len(PSNR_final_result_att_output_ori_input[name])
        perceptual_result_att_output_ori_input[name]=sum(perceptual_result_att_output_ori_input[name])/len(perceptual_result_att_output_ori_input[name])
        mse_final_result_att_output_ori_input[name]=sum(mse_final_result_att_output_ori_input[name])/len(mse_final_result_att_output_ori_input[name])
        edge_mse_final_result_ori_output_ori_input[name]=sum(edge_mse_final_result_ori_output_ori_input[name])/len(edge_mse_final_result_ori_output_ori_input[name])

        SSIM_final_result_att_output_ori_input_wo_logo[name]=sum(SSIM_final_result_att_output_ori_input_wo_logo[name])/len(SSIM_final_result_att_output_ori_input_wo_logo[name])
        PSNR_final_result_att_output_ori_input_wo_logo[name]=sum(PSNR_final_result_att_output_ori_input_wo_logo[name])/len(PSNR_final_result_att_output_ori_input_wo_logo[name])
        perceptual_result_att_output_ori_input_wo_logo[name]=sum(perceptual_result_att_output_ori_input_wo_logo[name])/len(perceptual_result_att_output_ori_input_wo_logo[name])
        mse_final_result_att_output_ori_input_wo_logo[name]=sum(mse_final_result_att_output_ori_input_wo_logo[name])/len(mse_final_result_att_output_ori_input_wo_logo[name])
        edge_mse_final_result_ori_output_ori_input_wo_logo[name]=sum(edge_mse_final_result_ori_output_ori_input_wo_logo[name])/len(edge_mse_final_result_ori_output_ori_input_wo_logo[name])

        SSIM_final_result_att_final_output_ori_final_output[name]=sum(SSIM_final_result_att_final_output_ori_final_output[name])/len(SSIM_final_result_att_final_output_ori_final_output[name])
        PSNR_final_result_att_final_output_ori_final_output[name]=sum(PSNR_final_result_att_final_output_ori_final_output[name])/len(PSNR_final_result_att_final_output_ori_final_output[name])
        perceptual_final_result_att_final_output_ori_final_output[name]=sum(perceptual_final_result_att_final_output_ori_final_output[name])/len(perceptual_final_result_att_final_output_ori_final_output[name])
        mse_final_result_att_final_output_ori_final_output[name]=sum(mse_final_result_att_final_output_ori_final_output[name])/len(mse_final_result_att_final_output_ori_final_output[name])
        edge_mse_final_result_att_final_output_ori_final_output[name]=sum(edge_mse_final_result_att_final_output_ori_final_output[name])/len(edge_mse_final_result_att_final_output_ori_final_output[name])

        if 'random_mask' in opt.algorithm:
            random_mask_SSIM_final_result[name]=sum(random_mask_SSIM_final_result[name])/len(random_mask_SSIM_final_result[name])
            random_mask_PSNR_final_result[name]=sum(random_mask_PSNR_final_result[name])/len(random_mask_PSNR_final_result[name])
            random_mask_perceptual_result[name]=sum(random_mask_perceptual_result[name])/len(random_mask_perceptual_result[name])
            random_mask_mse_final_result[name]=sum(random_mask_mse_final_result[name])/len(random_mask_mse_final_result[name])
            

        print(SSIM_final_result[name],PSNR_final_result[name],perceptual_result[name],mse_final_result[name])
        df['models'].append(name)
        df['SSIM_final_result'].append(SSIM_final_result[name])
        df['PSNR_final_result'].append(PSNR_final_result[name])
        df['perceptual_result'].append(perceptual_result[name])
        df['mse_final_result'].append(mse_final_result[name])
        df['edge_mse_final_result'].append(edge_mse_final_result[name])
        df['mask_mse_result'].append(mask_mse_result[name])

        df['SSIM_final_result_att_output_ori_input'].append(SSIM_final_result_att_output_ori_input[name])
        df['PSNR_final_result_att_output_ori_input'].append(PSNR_final_result_att_output_ori_input[name])
        df['perceptual_result_att_output_ori_input'].append(perceptual_result_att_output_ori_input[name])
        df['mse_final_result_att_output_ori_input'].append(mse_final_result_att_output_ori_input[name])
        df['edge_mse_final_result_ori_output_ori_input'].append(edge_mse_final_result_ori_output_ori_input[name])

        df['SSIM_final_result_att_output_ori_input_wo_logo'].append(SSIM_final_result_att_output_ori_input_wo_logo[name])
        df['PSNR_final_result_att_output_ori_input_wo_logo'].append(PSNR_final_result_att_output_ori_input_wo_logo[name])
        df['perceptual_result_att_output_ori_input_wo_logo'].append(perceptual_result_att_output_ori_input_wo_logo[name])
        df['mse_final_result_att_output_ori_input_wo_logo'].append(mse_final_result_att_output_ori_input_wo_logo[name])
        df['edge_mse_final_result_ori_output_ori_input_wo_logo'].append(edge_mse_final_result_ori_output_ori_input_wo_logo[name])

        df['SSIM_final_result_att_final_output_ori_final_output'].append(SSIM_final_result_att_final_output_ori_final_output[name])
        df['PSNR_final_result_att_final_output_ori_final_output'].append(PSNR_final_result_att_final_output_ori_final_output[name])
        df['perceptual_final_result_att_final_output_ori_final_output'].append(perceptual_final_result_att_final_output_ori_final_output[name])
        df['mse_final_result_att_final_output_ori_final_output'].append(mse_final_result_att_final_output_ori_final_output[name])
        df['edge_mse_final_result_att_final_output_ori_final_output'].append(edge_mse_final_result_att_final_output_ori_final_output[name])

        if 'random_mask' in opt.algorithm:
            df['random_mask_SSIM_final_result'].append(random_mask_SSIM_final_result[name])
            df['random_mask_PSNR_final_result'].append(random_mask_PSNR_final_result[name])
            df['random_mask_perceptual_result'].append(random_mask_perceptual_result[name])
            df['random_mask_mse_final_result'].append(random_mask_mse_final_result[name])
    f=open(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,name+'_maskded_only_result_for_wr.json'),'w')
    json.dump(df,f,indent=4)
    # with open(os.path.join('inpainting/ensemble/experiment_result/',opt.output_dir,name,name+'_result.json'),'w') as f:
    #     json.dump(df,f,indent=4)
    df=pd.DataFrame(df)
    df.to_excel(writer,sheet_name='Sheet1',index=False)
    writer.save()



if __name__=='__main__':
    main()