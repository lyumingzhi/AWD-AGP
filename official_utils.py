import torch
import cv2
import time
import time

import torch
from matplotlib import pyplot as plt

from inpainting.superpixel_fcn.train_util import update_spixl_map,shift9pos,get_spixel_image
from inpainting.superpixel_fcn import models 

import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from scipy.interpolate import LinearNDInterpolator
import numpy as np
# from pykeops.torch import LazyTensor


def position_variance(position_dict):

    value_list=np.array([ value for model, value in position_dict.items()])
    mean_position=np.mean(value_list,axis=0)
    distance_to_mean=np.sqrt(np.square(value_list[0,...]-mean_position[0])+np.square(value_list[1,...]-mean_position[1]))

    var=np.mean(distance_to_mean)
    return var

def rbf_interpolation(score,map):
    dtype = torch.cuda.FloatTensor 

    x_index=torch.linspace(0,1,map.size()[-1]).cuda()
    y_index=torch.linspace(0,1,map.size()[-2]).cuda()

    y_index,x_index=torch.meshgrid(y_index,x_index,indexing='ij')

    x_index=torch.masked_select(x_index,map>0)
    y_index=torch.masked_select(y_index,map>0)

    x=torch.stack((y_index,x_index),dim=-1)

    cv2.imwrite('inpainting/ensemble/score_map.jpg',(torch.abs(score)/torch.max(torch.abs(score))).detach().cpu().numpy()*255)
    score=torch.masked_select(score,map>0).unsqueeze(-1)
    # print('score',score.size())


    def laplacian_kernel(x, y, sigma=0.1):
        x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
        y_j = LazyTensor(y[None, :, :])  # (1, N, 1)

        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
        return (-D_ij.sqrt() / sigma).exp()  # (M, N) symbolic Laplacian kernel matrix
    

    alpha = 10  # Ridge regularization

    start = time.time()

    K_xx = laplacian_kernel(x, x)

    a = K_xx.solve(score, alpha=alpha)

    end = time.time()


    Y=torch.linspace(0,1,map.size()[-2]).cuda()
    X=torch.linspace(0,1,map.size()[-1]).cuda()
    Y, X = torch.meshgrid(Y, X,indexing='ij')
    t = torch.stack((Y.contiguous().view(-1), X.contiguous().view(-1)), dim=-1)
    t=t.view(-1,2)
 
    K_tx = laplacian_kernel(t, x)


    mean_t = K_tx @ a
    mean_t=mean_t.view(map.size())
    cv2.imwrite('inpainting/ensemble/interpolated_score_map.jpg',(torch.abs(mean_t)/torch.max(torch.abs(mean_t))).detach().cpu().numpy()*255)
    return mean_t.view(map.size()[-2],map.size()[-1])
    cv2.imwrite('inpainting/ensemble/meant.jpg',mean_t.detach().cpu().numpy()*255)

def ND_interp(score,map):
    # dtype = torch.cuda.FloatTensor 

    score1=torch.abs(score).cpu().numpy()
    
    # cv2.imwrite('inpainting/ensemble/score_map_SBS.jpg',((score1-np.min(score1))/(np.max(score1)-np.min(score1)))*255)

    x_index=torch.linspace(0,1,map.size()[-1]).cuda()
    y_index=torch.linspace(0,1,map.size()[-2]).cuda()

    y_index,x_index=torch.meshgrid(y_index,x_index,indexing='ij')

    
    x_index=torch.masked_select(x_index,map>0)
    y_index=torch.masked_select(y_index,map>0)

    x_index=x_index.cpu().numpy()
    y_index=y_index.cpu().numpy()

    # x=torch.stack((y_index,x_index),dim=-1)

    score=torch.masked_select(score,map>0).unsqueeze(-1)
    score=score.cpu().numpy().reshape((-1,))

    print(x_index.shape,y_index.shape,score.shape)
    # interp_func=SmoothBivariateSpline(x_index,y_index,score,s=0.0)
    interp_func=LinearNDInterpolator(list(zip(x_index,y_index)),score,fill_value=0)
    print(list(zip(x_index,y_index)))
    # exit()
    # X, Y = torch.meshgrid(X, Y)
    Y=torch.linspace(0,1,map.size()[-2])
    X=torch.linspace(0,1,map.size()[-1])
    
    # Y, X = torch.meshgrid(Y, X,indexing='ij')
    # t = torch.stack((Y.contiguous().view(-1), X.contiguous().view(-1)), dim=-1)
    # t=t.view(-1,2)

    X=X.cpu().numpy()
    Y=Y.cpu().numpy()
    X,Y=np.meshgrid(X,Y) #linearInterpolation
    interp_result=interp_func(X,Y)
    interp_result=np.abs(interp_result)
    print(np.sum(interp_result))
    
    inter_result_to_show=np.concatenate((((interp_result-np.min(interp_result))/(np.max(interp_result)-np.min(interp_result))),(np.abs(score1)-np.min(np.abs(score1)))/(np.max(np.abs(score1))-np.min(np.abs(score1)))),axis=-1)
    # cv2.imwrite('inpainting/ensemble/interpolated_score_map_Linear.jpg',inter_result_to_show*255)
    # exit()
    interp_result=torch.from_numpy(interp_result)
    return interp_result,inter_result_to_show

def get_semantic_loss(superpixel_map,mask,w1=100,w2=80,w3=60):

    # only care about superpixel map inside the mask
    masked_superpixel_map=mask*superpixel_map
    diversity=torch.unique(masked_superpixel_map)
    diversity_score=diversity.size()[0]-1

    ######################## get the number of different superpixels covered by the input mask #####################
    h_shift_unit=1
    w_shift_unit=1
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = F.pad(mask, (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit), mode='replicate')
    # input_pd = torch.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:,:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:,:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:,:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:,:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:,:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:,:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:,:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:,:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = torch.cat([     top_left,    top,      top_right,
                                        left,              right,
                                        bottom_left, bottom,    bottom_right], dim=1)


    shift=torch.sum(torch.abs(shift_tensor-center.repeat(1,8,1,1)),dim=1)
    # assert cv2.imwrite('inpainting/ensemble/superpixel_shift_img.jpg', (shift>0)[0].detach().cpu().numpy()*255)

    # get the index of superpixels that are on the edges
    edge_diversity=torch.unique((shift>0)*masked_superpixel_map)
    edge_diversity_score=edge_diversity.size()[0]-1
    internal_diversity_score=diversity_score-edge_diversity_score

    # get the isolated area totally inside the mask
    internal_iso_area=masked_superpixel_map.repeat(1,edge_diversity.size()[0],1,1)-edge_diversity.view(1,-1,1,1).repeat(1,1,*(masked_superpixel_map.size()[-2:]))
    internal_iso_area=torch.prod((internal_iso_area!=0),dim=1,keepdim=True)
    internal_iso_area=torch.ones(internal_iso_area.size()).cuda()*(internal_iso_area!=0)

    
    masked_superpixel_map+=0.3*(masked_superpixel_map>0)

    # get the ratio of isolation area over mask area
    internal_iso_area_score=torch.sum(internal_iso_area)/torch.sum(mask)

    # return the final score and use negative sign since we want to maximize this score to let masks cover more isolated superpixels as possible
    final_score=internal_iso_area_score
    return -final_score

# get the graph (relationship between among superpixel)
def get_superpixel_graph(superpixel_map):
    h_shift_unit=1
    w_shift_unit=1
    # input should be padding as (c, 1+ height+1, 1+width+1)
    
    input_pd = F.pad(superpixel_map.float(), (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit), mode='replicate').int()
    # input_pd = torch.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:,:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:,:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:,:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:,:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:,:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:,:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:,:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:,:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = torch.cat([     top_left,    top,      top_right,
                                        left,              right,
                                        bottom_left, bottom,    bottom_right], dim=1)
    
    # shift=torch.sum(torch.abs(shift_tensor-center.repeat(1,8,1,1)),dim=1)
    
    # get the relationship between each superpixels
    superpixel_graph={}
    for superpixel_id in torch.unique(superpixel_map).tolist():
        
        superpixel_graph[superpixel_id]=torch.unique(torch.masked_select(shift_tensor,(superpixel_map==superpixel_id).repeat(1,8,1,1))).tolist()

        # remove the itself, and collect the neighbor superpixel
        if superpixel_id in superpixel_graph[superpixel_id]:
            superpixel_graph[superpixel_id].remove(superpixel_id)
    # print(superpixel_graph)
    return superpixel_graph

# get the attribute values for each superpixel
def get_superpixel_node(superpixel_map,image,superpixel_graph):

    assert superpixel_map.dim()==4 and image.dim()==4
    superpixel_nodes={}
    for superpixel_id in superpixel_graph:
        # print(image.size(),superpixel_map.size())
        pixels_of_superpixel_id=image*(superpixel_map==superpixel_id)
        variance,mean=torch.var_mean(torch.masked_select(image,(superpixel_map==superpixel_id)))
        size=torch.sum((superpixel_map==superpixel_id)).item() 
        # superpixel_nodes[superpixel_id]=torch.Tensor([mean,variance,size]).cuda()
        superpixel_nodes[superpixel_id]=[torch.Tensor([mean,variance]).cuda(),size]
    
    
    all_feature_vector=torch.stack([feature_vector[0] for i,feature_vector in superpixel_nodes.items()]) # get all the features/attribute values of superpixels


    inter_superpixel_var, inter_superpixel_mean = torch.var_mean(all_feature_vector, dim=0) # cal the distribution of features of all superpixels

    normalized_all_feature_vector=(all_feature_vector-inter_superpixel_mean)/torch.sqrt(inter_superpixel_var) # nromalize the feature for each superpixel

    
    # save it back to the superpxiel node
    for i, superpixel_id in enumerate(superpixel_nodes):
        superpixel_nodes[superpixel_id]=[normalized_all_feature_vector[i],superpixel_nodes[superpixel_id][1]]
        # [normalized feature, size]
    

    return superpixel_nodes


def compare_superpixel_distance(superpixel_id1, superpixel_id2, superpixel_graph, new_superpixel_graph, inv_superpixel_graph, superpixel_nodes):
    features1=[superpixel_nodes[superpixel_id] for superpixel_id in new_superpixel_graph[inv_superpixel_graph[superpixel_id1]]]
    features2=[superpixel_nodes[superpixel_id] for superpixel_id in new_superpixel_graph[inv_superpixel_graph[superpixel_id2]]]
    # print(new_superpixel_graph[inv_superpixel_graph[superpixel_id1]])
    # print(torch.stack([features[0] for features in features1]))
    combined_feature_vector1=torch.matmul(torch.Tensor([features[1] for features in features1]).cuda()/sum([features[1] for features in features1]),torch.stack([features[0] for features in features1]))

    combined_feature_vector2=torch.matmul(torch.Tensor([features[1] for features in features2]).cuda()/sum([features[1] for features in features2]),torch.stack([features[0] for features in features2]))

    # intercluster_var1, intercluster_mean1 = torch.var_mean(torch.stack([features[0] for features in features1]), dim=0)
    # intercluster_var2, intercluster_mean2 = torch.var_mean(torch.stack([features[0] for features in features2]), dim=0)


    # print('mse loss btw feature', F.mse_loss(combined_feature_vector1, combined_feature_vector2))
    return F.mse_loss(combined_feature_vector1, combined_feature_vector2)


def merge_close_superpixel(superpixel_map,superpixel_graph,superpixel_nodes,thres=0.1):
    new_superpixel_graph={}
    for superpixel_id, superpixel_neighbors in superpixel_graph.items():
        new_superpixel_graph[superpixel_id]=[superpixel_id]
    inv_new_superpixel_graph={}
    for superpixel_id, superpixel_neighbors in superpixel_graph.items():
        inv_new_superpixel_graph[superpixel_id]=superpixel_id

    for superpixel_id in superpixel_graph:
        for neighb_superpixel_id in superpixel_graph[superpixel_id]:

            # if the neighbor superpixel is not in the list of child of the parent of the current superpixel, or the current superpixel is not in the list of child of the parent of the neighbor superpixel 
            if neighb_superpixel_id not in new_superpixel_graph[inv_new_superpixel_graph[superpixel_id]] and superpixel_id not in new_superpixel_graph[inv_new_superpixel_graph[neighb_superpixel_id]]:
                
                # if the superpixel and its neighbor superpixel is close enough
                if compare_superpixel_distance(superpixel_id, neighb_superpixel_id, superpixel_graph, new_superpixel_graph, inv_new_superpixel_graph, superpixel_nodes) <thres: #use the original superpxiel to compare instead of the merged one.

                    # add the neighbor superpixel to its parent superpixel
                    new_superpixel_graph[inv_new_superpixel_graph[superpixel_id]].extend(new_superpixel_graph[inv_new_superpixel_graph[neighb_superpixel_id]])
                    
                    # remove the neigbhor superpixel from its parent
                    superpixel_id_to_remove=inv_new_superpixel_graph[neighb_superpixel_id]

                    # before removing the neighbor superpixel, set parent superpixel of all the child superpixel of the neighbor superpixel as the parent of current pixel. (merging)
                    for child_superpixel_id in new_superpixel_graph[superpixel_id_to_remove]:
                        inv_new_superpixel_graph[child_superpixel_id]=inv_new_superpixel_graph[superpixel_id]
                    
                    # remove the neighbor superpixel, since it has been merged
                    new_superpixel_graph[superpixel_id_to_remove]=[]
                    
    # get the new merged superpixel and get the number of pixels in each merged superpixel in the superpixel map
    merged_superpixel_map=torch.zeros(superpixel_map.size()).cuda()
    num_pixel=0
    for i, superpixel_id in enumerate(new_superpixel_graph):
        for child_superpixel_id in new_superpixel_graph[superpixel_id]:
            merged_superpixel_map+=torch.ones(merged_superpixel_map.size()).cuda()*i*(superpixel_map==child_superpixel_id)
            num_pixel+=torch.sum(superpixel_map==child_superpixel_id)

    # get the edge of the superpixels
    h_shift_unit=1
    w_shift_unit=1
    input_pd = F.pad(merged_superpixel_map.float(), (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit), mode='replicate').int()
    # input_pd = torch.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:,:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:,:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:,:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:,:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:,:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:,:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:,:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:,:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = torch.cat([     top_left,    top,      top_right,
                                        left,              right,
                                        bottom_left, bottom,    bottom_right], dim=1)
    
    shift=torch.sum(torch.abs(shift_tensor-center.repeat(1,8,1,1)),dim=1)
    edge_image=shift>0

    return merged_superpixel_map, edge_image
    

def get_superpixel(image):

    downsize=16
      # Data loading code
    input_transform = transforms.Compose([
        # flow_transforms.ArrayToTensor(),
        # transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])
    
    
    # to get superpixels of the images, we use a pretrained model
    pretrained='./inpainting/superpixel_fcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar'
    network_data = torch.load(pretrained)
    # print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']]( data = network_data).cuda()
    model.eval()

    cudnn.benchmark = True


    img_=image
    

    H, W= img_.size()[-2:]
    H_, W_  = int(np.ceil(H/16.)*16), int(np.ceil(W/16.)*16)


    # get spixel id. Resolution of the superpixel map is downscaled by downsize
    n_spixl_h = int(np.floor(H_ / downsize))
    n_spixl_w = int(np.floor(W_ / downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values) # get the shifted coordinators of pixels on 9 directions
    
    # resize the superpixel id map back to the original size of images
    spix_idx_tensor = np.repeat(
      np.repeat(spix_idx_tensor_, downsize, axis=1), downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

    n_spixel =  int(n_spixl_h * n_spixl_w)


    # img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img = F.interpolate(img_, size=( H_,W_), mode='bicubic')[0]
    img_=img_[0]
    
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # compute superpixel output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))
    toc = time.time() - tic
    
    # output=merge_close_superpixel(output,None)

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)

    # resize it to the size of images
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(image.size()[-2:]), mode='nearest').type(torch.int) + 1

    # get the superpixel map visuable
    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1).cuda()
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= n_spixel,  b_enforce_connect=True)

    # merge the closest superpixel together according to the distribution of the pixels inside the superpixel
    old_ori_sz_pixel_map=ori_sz_spixel_map
    edge=None
    while True:
        old_ori_sz_pixel_map=ori_sz_spixel_map
        
        # get the adjecant relationship between each pair of superpixel
        superpixel_graph=get_superpixel_graph(ori_sz_spixel_map)
        # get the feature for each superpixel in the current superpixel map
        superpixel_nodes=get_superpixel_node(ori_sz_spixel_map,image,superpixel_graph)
        # merge the superpixels that are adjcent and have close feature
        ori_sz_spixel_map, edge=merge_close_superpixel(ori_sz_spixel_map,superpixel_graph,superpixel_nodes)
        if torch.sum(torch.abs(old_ori_sz_pixel_map - ori_sz_spixel_map))==0:
            break

    spixel_viz=torch.clamp(image+edge,min=0,max=1)[0].detach().cpu().numpy()
    return ori_sz_spixel_map, spixel_viz

    
if __name__=='!__main__':
    import time

    import torch
    from matplotlib import pyplot as plt

    from pykeops.torch import LazyTensor

    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    N = 10000 if use_cuda else 1000  # Number of samples

    # # Sampling locations:
    # x = torch.rand(N, 1).type(dtype)

    # # Some random-ish 1D signal:
    # b = (
    #     x
    #     + 0.5 * (6 * x).sin()
    #     + 0.1 * (20 * x).sin()
    #     + 0.05 * torch.randn(N, 1).type(dtype)
    # )

    # Sampling locations:
    x = torch.rand(N, 2).type(dtype)

    # Some random-ish 2D signal:
    b = ((x - 0.5) ** 2).sum(1, keepdim=True)
    b[b > 0.4**2] = 0
    b[b < 0.3**2] = 0
    b[b >= 0.3**2] = 1
    b = b + 0.05 * torch.randn(N, 1).type(dtype)

    # Add 25% of outliers:
    Nout = N // 4
    b[-Nout:] = torch.rand(Nout, 1).type(dtype)
   
    #####################################################
    map=torch.zeros((256,256))
    index=(torch.rand((2,10000))*map.size()[-1]).long()

    b=torch.rand((10000,1)).cuda()
    x=index.transpose(1,0).cuda()/map.size()[-1]
    
        
    score=torch.zeros((256,256))
    for i in range(index.size()[-1]):
        map[index[0,i],index[1,i]]=1
        score[index[0,i],index[1,i]]=i/index.size()[-1]
    ############################################

    def laplacian_kernel(x, y, sigma=0.1):
        x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
        y_j = LazyTensor(y[None, :, :])  # (1, N, 1)

        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
        return (-D_ij.sqrt() / sigma).exp()  # (M, N) symbolic Laplacian kernel matrix
    

    alpha = 10  # Ridge regularization

    start = time.time()

    K_xx = laplacian_kernel(x, x)

    a = K_xx.solve(b, alpha=alpha)

    end = time.time()

    print(
        "Time to perform an RBF interpolation with {:,} samples in 2D: {:.5f}s".format(
            N, end - start
        )
    )
    # Extrapolate on a uniform sample:
    X = Y = torch.linspace(0, 1, 101).type(dtype)
    X, Y = torch.meshgrid(X, Y)
    t = torch.stack((X.contiguous().view(-1), Y.contiguous().view(-1)), dim=1)

    K_tx = laplacian_kernel(t, x)

    print(K_tx.shape,a.shape)
    mean_t = K_tx @ a
    mean_t = mean_t.view(101, 101)

    print(mean_t)
    cv2.imwrite('inpainting/ensemble/meant.jpg',mean_t.detach().cpu().numpy()*255)
    exit()

    # 2D plot: noisy samples and interpolation in the background
    plt.figure(figsize=(8, 8))

    plt.scatter(
        x.cpu()[:, 0], x.cpu()[:, 1], c=b.cpu().view(-1), s=25000 / len(x), cmap="bwr"
    )
    plt.imshow(
        mean_t.cpu().numpy()[::-1, :],
        interpolation="bilinear",
        extent=[0, 1, 0, 1],
        cmap="coolwarm",
    )

    # sphinx_gallery_thumbnail_number = 2
    plt.axis([0, 1, 0, 1])
    plt.tight_layout()
    # plt.show()

    plt.savefig('inpainting/ensemble/pykeop.png')
        
        
        
if __name__=='!__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import SmoothBivariateSpline

    import warnings
    warnings.simplefilter('ignore')


    train_x, train_y = torch.meshgrid(torch.arange(-5, 5, 0.5), torch.arange(-5, 5, 0.5))
    # print(train_x)

    train_x, train_y = np.meshgrid(np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5))
    # print(train_y)
    
    train_x = train_x.flatten()
    train_y = train_y.flatten()
    
    # def z_func(x, y):
    #     return np.cos(x) + np.sin(y) ** 2 + 0.05 * x + 0.1 * y
    def z_func(x,y):
        return y

    train_z = z_func(train_x, train_y)
    print('train_x',train_x.shape,train_z.shape)
    interp_func = SmoothBivariateSpline(train_x, train_y, train_z, s=0.0)
    smth_func = SmoothBivariateSpline(train_x, train_y, train_z)

    test_x = np.arange(-9, 9, 0.01)
    test_y = np.arange(-9, 9, 0.01)
    grid_x, grid_y = np.meshgrid(test_x, test_y)

    print('tesst',test_x.shape)
    interp_result = interp_func(test_x, test_y)
    print('interp result.',interp_result.shape)
    smth_result = smth_func(test_x, test_y).T
    perfect_result = z_func(grid_x, grid_y)
    perfect_result[perfect_result<0]=0
    assert cv2.imwrite('perfect_result.jpg',(perfect_result-np.min(perfect_result))/(np.max(perfect_result)-np.min(perfect_result))*255)
    assert cv2.imwrite('interp_result.jpg',(interp_result-np.min(interp_result))/(np.max(interp_result)-np.min(interp_result))*255)

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    extent = [test_x[0], test_x[-1], test_y[0], test_y[-1]]
    opts = dict(aspect='equal', cmap='nipy_spectral', extent=extent, vmin=-1.5, vmax=2.5)

    im = axes[0].imshow(perfect_result, **opts)
    fig.colorbar(im, ax=axes[0], orientation='horizontal')
    axes[0].plot(train_x, train_y, 'w.')
    axes[0].set_title('Perfect result, sampled function', fontsize=21)

    im = axes[1].imshow(smth_result, **opts)
    axes[1].plot(train_x, train_y, 'w.')
    fig.colorbar(im, ax=axes[1], orientation='horizontal')
    axes[1].set_title('s=default', fontsize=21)

    im = axes[2].imshow(interp_result, **opts)
    fig.colorbar(im, ax=axes[2], orientation='horizontal')
    axes[2].plot(train_x, train_y, 'w.')
    axes[2].set_title('s=0', fontsize=21)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    from inpainting.ensemble.options import get_args
    from inpainting.ensemble.dataset import AttackDataset,TransferAttackDataset
    from torch.utils.data import DataLoader


    opt=get_args()
    dataset=AttackDataset(opt,opt.InputImg_dir,relative_mask_size=0.33,if_sort=True)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    for index,(img, mask,rect,target_patch, logo) in enumerate(iter(dataloader)):
    
        # img=cv2.imread('/home1/mingzhi/inpainting/crfill/datasets/places2sample1k_val/places2samples1k_crop256/Places365_val_00010293.jpg')
        # mask=cv2.imread('/home1/mingzhi/inpainting/ensemble/experiment_result/pred_masks_from_latest_fasterrcnn_mask_instance_mixture/373.png')
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # img=transforms.ToTensor()(img).unsqueeze(0).cuda()
        # mask=transforms.ToTensor()(mask).unsqueeze(0).cuda()

        img = img.cuda()
        graph,vis_graph=get_superpixel(img)

        assert cv2.imwrite('inpainting/ensemble/superpixel_maps/superpixel_graph_example'+str(index)+'.png', vis_graph.transpose(1,2,0)*255)
        # exit()

    # smantic_loss = get_semantic_loss(graph,mask )

    # print(smantic_loss)