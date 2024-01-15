from inpainting.ensemble.mask_RPN.RPN_trainer import FasterRCNNTrainer
from inpainting.ensemble.mask_RPN.RPN_dataset import RPN_dataset
from inpainting.ensemble.mask_RPN.RPN_net import FasterRCNNVGG16
from torch.utils import data as data_
# from inpainting.ensemble.mask_RPN.config import opt
from tqdm import tqdm
import torch.nn.functional as F
import os
from inpainting.simple_faster_rcnn_pytorch.utils import array_tool as at
from inpainting.simple_faster_rcnn_pytorch.utils.vis_tool import visdom_bbox
from inpainting.ensemble.mask_RPN.config_RPNnet import Config
import torch

from inpainting.ensemble.mask_RPN.RPN_trainer import _fast_rcnn_loc_loss
import numpy as np
import cv2
import torchvision.transforms as transforms
from inpainting.ensemble.official_dataset import AttackDataset,TransferAttackDataset
import json

def inference(**kwargs):
    opt = Config()
    opt._parse(kwargs)
    seed = 1
    torch.cuda.manual_seed(seed)


    # dataset=AttackDataset(opt,'/home1/mingzhi/inpainting/crfill/datasets/places2sample1k_val/places2samples1k_crop256',relative_mask_size=0.33,if_sort=True)
    dataset=AttackDataset(opt,'/home1/mingzhi/inpainting/Watermarking',relative_mask_size=0.33,if_sort=True)

    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  # pin_memory=True,``
                                  num_workers=1)
    RPN_net=FasterRCNNVGG16(bbox_preset_size=int(256*0.33), opt=opt)
    print('model construction complete')

    trainer = FasterRCNNTrainer(RPN_net, opt).cuda()

    trainer.load('/home1/mingzhi/inpainting/ensemble/checkpoints/latest_fasterrcnn_mask_instance_mixture')
    opt.caffe_pretrain=True

    pred_mask = {'pred_mask':[]}

    os.makedirs(opt.output_dir, exist_ok= True)
    for i, (img, mask, rect, target_img, _) in enumerate(dataloader):
        # img = cv2.imread('/home1/mingzhi/inpainting/crfill/datasets/places2sample1k_val/places2samples1k_crop256/Places365_val_00000049.jpg')
        # img=transforms.ToTensor()(img).unsque6eze(0)
        img = img[:,[2,1,0],...] #wrong
        

        _bboxes, _labels, _scores, _anchors, _loc, _final_keep_indicators = trainer.faster_rcnn.predict(img, visualize=True)

        keep_mask=np.zeros((_bboxes[0].shape[0],))
        keep_mask[_final_keep_indicators[0][0].detach().cpu().numpy()]=True
        # gt_rpn_label_of_proposed_rois = trainer.get_gt_pred_label(img, _bboxes[0].detach().cpu().numpy(), keep_mask)

        
        pred_img = visdom_bbox( img[0].detach().cpu().numpy()*255,
            at.tonumpy(_bboxes[0][_final_keep_indicators[0][0]]),
            # gt_score = at.tonumpy(gt_rpn_label_of_proposed_rois[_final_keep_indicators[0][0]]).reshape(-1),
            score = at.tonumpy(_scores[0][_final_keep_indicators[0][0]]))
        # assert cv2.imwrite(os.path.join('/home1/mingzhi/inpainting/ensemble/mask_RPN/test_results/', str(i)+'.png'), pred_img.transpose(1,2,0)[...,::-1]*255)
        # trainer.vis.img('pred_img', pred_img)

        pred_bbox = _bboxes[0][_final_keep_indicators[0][0]]
        # if (pred_bbox[:, -2:] - pred_bbox[:,:2])[0,0]<84 or (pred_bbox[:, -2:] - pred_bbox[:,:2])[0,1]< 84:
        #     print('pred_bbox',pred_bbox)
        #     exit(0)
        mask,tmp_rect=trainer.generate_rect_mask(img.size()[-2:],mask_size= (pred_bbox[:, -2:].int() - pred_bbox[:,:2].int()).detach().cpu().numpy(),position=(pred_bbox[:,:2].detach().cpu().numpy()).astype(int),rand_mask=False,batch_size=img.size()[0])

        pred_mask['pred_mask'].append(pred_bbox[0].tolist())
        assert cv2.imwrite(os.path.join(opt.output_dir, str(i)+'.png'), mask[0,0].transpose(1,2,0)*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    with open(os.path.join(opt.output_dir,'rect.json'),'w') as f:
        json.dump(pred_mask, f, indent = 4)
        
if __name__ =='__main__':
    inference()