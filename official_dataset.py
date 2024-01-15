from unicodedata import name
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import os
import cv2
import torchvision.transforms as transforms
import numpy as np
from os.path import exists
import random
import re
import json
import torchvision

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class AttackDataset(torchvision.datasets.ImageFolder):
    def __init__(self,opt,img_dir,relative_mask_size=0.33,if_sort=False, transform=None, target_transform=None):
        self.opt=opt
        # self.img_dir=img_dir
        # self.file_list=[]
        # for root,dirs,files in os.walk(self.img_dir):
        #     for file in files:
        #         self.file_list.append(file)
        
        # if if_sort:
        #     self.file_list.sort(key=lambda f: int(re.sub('\D', '', f)))

        # self.filename_list=[filename.strip('.')[:-4] for filename in self.file_list]
        
        # self.file_list=[os.path.join(self.img_dir,filename) for filename in self.file_list]

        self.transformer =transforms.Compose([
                            transforms.ToTensor(),
                            transforms.CenterCrop((256,256)),
                            transforms.Resize((256,256))
                        ])
        self.mask_transformer =transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((256,256))
                        ])

        super().__init__(img_dir, transform=self.transformer, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions=IMG_EXTENSIONS, is_valid_file=None)

        # self.loader = None
        # self.extensions = None

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
       

        self.rect_file = None
        if self.opt.RPNRefineMask != None:
            with open(os.path.join(self.opt.RPNRefineMask, 'rect.json')) as f:
                self.rect_file = json.load(f)
 
            
        print('file list', len(self.samples))

        if  'random_logo_alpha' in self.opt.algorithm:
            with open('inpainting/ensemble/random_logo_alpha.json') as f:
                self.watermark_alpha_list = json.load(f)
        self.relative_mask_size=relative_mask_size
        

    def __len__(self):
        return len(self.samples)
    
    def generate_rect_mask(self,im_size, mask_size, margin=8, rand_mask=False, position = 'center'):

        # initialize the mask with a default size if there is not input mask size
        if mask_size==None:
            mask_size=(np.random.randint(int(0.2*im_size[0]),int(0.5*im_size[0])),int(0.2*im_size[1]),int(0.5*im_size[1]))
        mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)


        if rand_mask:
            # randomly initialize the mask
            sz0, sz1 = mask_size[0], mask_size[1]
            of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
            of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
        else:
            if position == 'center':
                # initialize the mask at the center of the images
                sz0, sz1 = mask_size[0], mask_size[1]
                of0 = (im_size[0] - sz0) // 2
                of1 = (im_size[1] - sz1) // 2
            elif position == 'left_top':
                # initialize the mask at the left top of the images
                sz0, sz1 = mask_size[0], mask_size[1]
                of0 = 0
                of1 = 0
            else:
                # initialize the mask at the input position
                of0 = max(position[0],0)
                of1 = max(position[2],0)
                sz0, sz1 = min(mask_size[0],im_size[0] - 1 - position[0]), min(mask_size[1],im_size[1] - 1 - position[2])
                

        mask[of0:of0+sz0, of1:of1+sz1] = 1
        mask = np.expand_dims(mask, axis=0)
        
        rect = torch.from_numpy(np.array([[of0, sz0, of1, sz1]], dtype=int))

        self.rect_list = []
        return mask, rect
    def __getitem__(self,index):
        
        # img_path=self.file_list[idx]
        # print(img_path)
        
        # img=cv2.imread(img_path) #bgr
        
        # load images
        path, target = self.samples[index]
        # img = self.loader(path)
        img=cv2.imread(path) #bgr

        # img = img[..., :: -1]  # reverse channel

        h, w= img.shape[0],img.shape[1]
        grid = 8
        img = img[:h//grid*grid, :w//grid*grid,:]
        # mask = mask[:,:,:h//grid*grid, :w//grid*grid]
        
        # img=self.transform(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # generate mask
        if self.opt.target_attack:
            mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )), (int(img.size()[-2]*self.relative_mask_size),int(img.size()[-1]*self.relative_mask_size)),rand_mask=False) #relative size=0.3

        # load mask from the predefined mask directory
        elif self.opt.RPNRefineMask != None:
            ''' mask = cv2.imread(os.path.join(self.opt.RPNRefineMask, str(idx)+'.png'))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.expand_dims((mask).astype('float')/255 > 0.5, axis = 0)'''
            rect = np.array(self.rect_file['pred_mask'][index]).astype('int')
            # rect = np.array([rect.item(0), rect.item(2) - rect.item(0), rect.item(1), rect.item(3) - rect.item(1)])

            # these lines can adjust size of watermark box########################################### 
            mid_x = int((rect.item(0) + rect.item(1)/2)) # 77
            mid_y = int((rect.item(2) + rect.item(3)/2))
            # length_x = int((rect.item(2) - rect.item(0)) * self.opt.relative_mask_size)//2*2
            # length_y = int((rect.item(3) - rect.item(1)) * self.opt.relative_mask_size)//2*2
            length_x = int(img.size()[-2]*self.opt.relative_mask_size)
            length_y = int(img.size()[-1]*self.opt.relative_mask_size)

            if rect[0] == 0:
                f0 = 0
            else:
                f0 = int(mid_x - length_x//2)
            if rect[2] == 0:
                f1 = 0
            else:
                f1 = int(mid_y - length_y//2)
            if f0 + rect[1] >= img.size()[-2]-1:
                f0 = img.size()[-2] - 1 - int(length_x)
            s0 = int(length_x)
            if f1 + rect[3] >= img.size()[-1]-1:
                f1 = img.size()[-1] - 1 - int(length_y)
            s1 = int(length_y)
            
            rect = np.array([f0, s0, f1, s1])
            
            mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )), (s0,s1),rand_mask=False, position=rect) #array([126,  84, 173,  84]) #array([126,  84, 175,  81])
            rect = rect.numpy()
            rect = np.array([rect.item(0), rect.item(1), rect.item(2), rect.item(3)])

        else:    
            # mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )), None,rand_mask=self.opt.random_mask)
            # mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )), (int(img.size()[-2]*self.relative_mask_size),int(img.size()[-1]*self.relative_mask_size)),rand_mask=True)

            # it is not supposed to go to this branch
            if self.rect_file!=None:
                mask=cv2.imread(os.path.join('/home1/mingzhi/inpainting/ensemble/experiment_result/surrogate_Mat_random_mask_baseline_perceptual_loss_only_inpaintingModel_place21000_ensemble_w_ModelNorm_grad_iter=100_e=0.03_mask_fix=0.33_sign/Matnet','attacked_mask','mask__'+str(index)+'.png'))
                
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = np.expand_dims((mask).astype('float')/255 > 0.5, axis = 0)
           
                rect = np.array(self.rect_file[list(self.rect_file.keys())[0]][idx])
                rect = np.array([rect.item(0), rect.item(1), rect.item(2), rect.item(3)]).astype('int')

            # randomly initialized the mask
            elif 'random_mask' in self.opt.algorithm:
                # random_mask_size=random.choice([(int(256*0.33),int(256*0.33)),(100,50),(50,100),(200,100),(100,200)])
                random_mask_size = (int(256*0.33),int(256*0.33))
                mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )),random_mask_size,rand_mask=True)
                rect = rect.numpy()
                rect = np.array([rect.item(0), rect.item(1), rect.item(2), rect.item(3)])

            # initialize the mask on certain position
            else:
                mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )), (int(img.size()[-2]*self.relative_mask_size),int(img.size()[-1]*self.relative_mask_size)),rand_mask=False, position='left_top')

                rect = rect.numpy()
                rect = np.array([rect.item(0), rect.item(1), rect.item(2), rect.item(3)])
            
        # self.rect_list.append(rect.tolist())

        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        # print('mask shape', mask.shape)
        # exit()

        if self.opt.target_attack and self.opt.target_img!='':
            target_img=cv2.imread(self.opt.target_img)
            target_img=self.transform(target_img)
        else:
            target_img=img
        
        logo = 0
        
        # get the logo to embed to the images
        if self.opt.get_logo == True:
 
            logo = cv2.imread(os.path.join('inpainting/logo/train_logo', str(index%44 +1)+'.png'), cv2.IMREAD_UNCHANGED)
            logo_alpha = logo[..., -1]
            logo = logo[..., :3]

            logo = self.mask_transformer(logo)
            logo_alpha = self.mask_transformer(logo_alpha)
            logo_alpha = transforms.Resize((rect.item(1), rect.item(3)))(logo_alpha).repeat(3,1,1)
            logo = transforms.Resize((rect.item(1), rect.item(3)))(logo)

            # embed the loaded watermark to the images
            if self.opt.attach_logo == True:
                if self.watermark_alpha_list!=None:
                    watermark_alpha = self.watermark_alpha_list[index]
                else:
                    watermark_alpha = 0.1

                img[...,rect.item(0):rect.item(0)+rect.item(1), rect.item(2):rect.item(2)+rect.item(3)] = img[...,rect.item(0):rect.item(0)+rect.item(1), rect.item(2):rect.item(2)+rect.item(3)]*(1- watermark_alpha*logo_alpha.unsqueeze(0)) + (logo * watermark_alpha*logo_alpha.unsqueeze(0) )

         

        # rect [top, hight, left, width]
        return img, mask, rect, target_img, logo

# class TransferAttackDataset(Dataset):
class TransferAttackDataset(torchvision.datasets.ImageFolder):
    def __init__(self,opt,img_dir,experiment_dir,name_to_test, if_sort = True):
        self.opt=opt
        self.img_dir=img_dir
        self.experiment_dir=experiment_dir
        name_to_test = 'Matnet'
        self.name_to_test=name_to_test
        if 'DWV' in opt.lossType or 'IWV' in opt.lossType:
            # self.name_to_test = 'WDModel'
            self.name_to_test = 'DBWEModel'
        print('name to test', self.name_to_test)
        self.file_list=[]
        self.adv_file_list=[]
        index_example=0

        super().__init__(img_dir, transform=None, target_transform=None)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions=IMG_EXTENSIONS, is_valid_file=None)

        # advImgNum=0
        # for root,dirs,files in os.walk(os.path.join(self.experiment_dir,name_to_test,'attacked_input')):
        #     for file in files:
        #         advImgNum+=1


        # for root,dirs,files in os.walk(self.img_dir):
        #     for file in files:
                
        #         self.file_list.append(os.path.join(self.img_dir,file))

        # if if_sort:
        #     self.file_list.sort(key=lambda f: int(re.sub('\D', '', f)))

        self.file_list = [sample[0] for sample in samples]
     

        advImgNum=0

        # get the number of adversarial examples for the target model
        for root,dirs,files in os.walk(self.experiment_dir):
            for dir in dirs:
                # self.surrogate_model=dir
                if self.name_to_test == dir:
                    for root,dirs,files in os.walk(os.path.join(self.experiment_dir,dir,'attacked_input',)):
                        
                        for file in files:
                            # self.adv_file_list.append(os.path.join(self.experiment_dir,dir,'attacked_input', file))
                            advImgNum+=1
                        break
                    break
            break
        

        # load information for all the mask
        self.rect_file = None
        if self.experiment_dir != None:
            with open(os.path.join(self.experiment_dir, self.name_to_test, 'rect.json')) as f:
                self.rect_file = json.load(f)

        # get the path for all the adversarial examples for the target models
        for index_example in range((advImgNum)):
            self.adv_file_list.append(os.path.join(self.experiment_dir,self.name_to_test,'attacked_input', str(index_example)+'.png'))
                
        
        # if there is no adversarial examples, return None
        if self.adv_file_list==[]:
            return None
        

        
        # self.adv_file_list=self.adv_file_list
        # self.file_list=self.file_list
        
        if  'random_logo_alpha' in self.opt.algorithm or True :
            with open('inpainting/ensemble/random_logo_alpha.json') as f:
                self.watermark_alpha_list = json.load(f)


        self.transformer=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.CenterCrop((256,256)),
                            transforms.Resize((256,256))
                        ])
        self.mask_transformer=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((256,256))
                        ])
        # self.transformer=transformer=transforms.Compose([
        #                     transforms.ToTensor(),
        #                     # transforms.Resize((256,256))
        #                 ])
    def __len__(self):
        return min(len(self.file_list),len(self.adv_file_list))
    
    def generate_rect_mask(self,im_size, mask_size, margin=8, rand_mask=True):
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
    def __getitem__(self,idx):
        
        img_path=self.file_list[idx]
        
        
        img=cv2.imread(img_path)

        h, w= img.shape[0],img.shape[1]
        grid = 8
        img = img[:h//grid*grid, :w//grid*grid,:]
        # mask = mask[:,:,:h//grid*grid, :w//grid*grid]

        img=self.transformer(img)
        

        # usually go to this branch, and load the mask images and its information (coordination of the mask)
        if 'optimal_mask_search' in self.opt.algorithm or True:
            mask=cv2.imread(os.path.join(self.experiment_dir,self.name_to_test,'attacked_mask','mask__'+str(idx)+'.png'))
            
            mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).reshape(mask.shape[0],mask.shape[1],1)
 
            rect = np.array(self.rect_file[list(self.rect_file.keys())[0]][idx])
            rect = np.array([rect.item(0), rect.item(1), rect.item(2), rect.item(3)]).astype('int')
            """ if rect.size == 1:
                rect = np.array([rect[0], rect[2] - rect[0], rect[1], rect[3] - rect[1]])
            elif rect.size ==2:
                rect = np.array([rect[0], rect[0,2] - rect[0,0], rect[0,1], rect[0,3] - rect[0,1]]) """
            # rect= torch.from_numpy(np.array([[0, 0, 0, 0]], dtype=int))
        elif self.opt.target_attack:
            mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )), (int(img.size()[-2]*0.3),int(img.size()[-1]*0.3 )),rand_mask=False)
            
        else:    
            # mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )), None,rand_mask=self.opt.random_mask)
            mask,rect=self.generate_rect_mask((int(img.size()[-2] ),int(img.size()[-1] )), (int(img.size()[-2]*0.9),int(img.size()[-1]*0.9 )),rand_mask=False)

        # randomly initialize the location of mask
        if 'random_mask' in self.opt.algorithm:
            random_mask=np.zeros(mask.shape)
            random_mask_size=100
       
            
            upper_left_x=random.randint(0,mask.shape[-1]-1-100)
            
            upper_left_y=random.randint(0,mask.shape[-2]-1-100)
 
            
            random_mask[:,upper_left_y:upper_left_y+random_mask_size,upper_left_x:upper_left_x+random_mask_size]=1
        

        # load adversarial examples
        # print(len(self.adv_file_list),idx)
        adv_img_path=self.adv_file_list[idx]
        
        
        adv_img=cv2.imread(adv_img_path)
        h,w=adv_img.shape[0], adv_img.shape[1]
        grid = 8
        adv_img = adv_img[:h//grid*grid, :w//grid*grid,:]
        # mask = mask[:,:,:h//grid*grid, :w//grid*grid]

        adv_img=self.transformer(adv_img)

        




        # mask = (torch.from_numpy(mask).type(torch.FloatTensor))
        mask = self.transformer(mask)
        assert mask.dim()==3 and mask.size()[0]==1
        if 'random_mask' in self.opt.algorithm:
            random_mask = (torch.from_numpy(random_mask).type(torch.FloatTensor))

        if self.opt.target_attack and self.opt.target_img!='':
            target_img=cv2.imread(self.opt.target_img)
            target_img=self.transformer(target_img)
        else:
            target_img=img
        
        img_wo_logo = img.clone()
        logo = 0

        # embed logo to clean images
        if self.opt.get_logo == True:

            logo = cv2.imread(os.path.join('inpainting/logo/train_logo', str(idx%44 +1)+'.png'), cv2.IMREAD_UNCHANGED)
            print(logo.shape)
            logo_alpha = logo[..., -1]
            logo = logo[..., :3]

            logo = self.mask_transformer(logo)
            logo_alpha = self.mask_transformer(logo_alpha)
            logo_alpha = transforms.Resize((rect.item(1), rect.item(3)))(logo_alpha).repeat(3,1,1)
            print('logo alpha',logo_alpha.shape)
            logo = transforms.Resize((rect.item(1), rect.item(3)))(logo)

            if self.opt.attach_logo == True:
                if self.watermark_alpha_list!=None:
                    watermark_alpha = self.watermark_alpha_list[idx]

                img[...,rect.item(0):rect.item(0)+rect.item(1), rect.item(2):rect.item(2)+rect.item(3)] = img[...,rect.item(0):rect.item(0)+rect.item(1), rect.item(2):rect.item(2)+rect.item(3)]*(1- watermark_alpha*logo_alpha.unsqueeze(0)) + (logo * watermark_alpha*logo_alpha.unsqueeze(0) )

        
        if 'evaluate_rw' in self.opt.algorithm:
            return  img, adv_img, mask,rect, target_img, logo_alpha[0:1,...], img_wo_logo
        if 'random_mask' in self.opt.algorithm:
            return  img, adv_img, mask,rect, target_img, random_mask, logo
        else:
            return  img, adv_img, mask,rect, target_img, logo