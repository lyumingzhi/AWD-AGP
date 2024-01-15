import torch.nn
from torch.nn import parallel
from torch.nn.modules.module import Module
class Noise_Func(Module):
    def __init__(self,input_size,a=None,b=None,target_models=None,PGD=True):
        super(Noise_Func,self).__init__()
        self.a=None
        self.b=None
        self.target_models=target_models
        self.PDG=True

        if a!=None and b!=None:
            if len(input_size)==3:
                self.a=torch.nn.Parameter((torch.zeros((3,1,1))+a),requires_grad=True)
                self.b=torch.nn.Parameter((torch.zeros((3,1,1))+b),requires_grad=True)
            elif len(input_size)==4:
                self.a=torch.nn.Parameter((torch.zeros((1,3,1,1))+a),requires_grad=True)
                self.b=torch.nn.Parameter((torch.zeros((1,3,1,1))+b),requires_grad=True)
        
        if PGD==False:
            self.delta_x=torch.nn.Parameter(torch.zeros(input_size),requires_grad=True)
        else:
            self.delta_x=torch.nn.Parameter(torch.zeros(input_size)+torch.rand(input_size)*0.0001,requires_grad=True)
        
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,x,mask=None,inter_grad=None,inter_grad_target=['delta_x']):

        if self.a!=None and self.b!=None:
            return self.delta_x*mask+x*(1+self.a.expand(x.size())*0.1)+self.b.expand(x.size())*0.1
        
        adv_example_for_each_model={}
        
        if inter_grad!=None: inter_grad['attack']={}
        def print_grad(name,target):
            if inter_grad is None:
                def grad_func(grad):
                    pass
                return grad_func
            else:
                def grad_func(grad):
                    if target=='delta_x':
                        inter_grad['attack']['delta_x'][name]=grad
                    elif target=='mask':
                        inter_grad['attack']['mask'][name]=grad

                return grad_func
        for name in self.target_models.keys():
            # print(name,'to add noise')
            if mask!=None:
                # adv_example_for_each_model[name]=self.delta_x*mask+x
                if 'delta_x' in inter_grad_target:
                    if 'delta_x' not in inter_grad:
                        inter_grad['attack']['delta_x']={}
                    delta_x_clone=self.delta_x.clone()
                    delta_x_clone.register_hook(print_grad(name,'delta_x'))
                    if isinstance(x,dict):
                        adv_example_for_each_model[name]=delta_x_clone*mask+x[name]
                    else:
                        adv_example_for_each_model[name]=delta_x_clone*mask+x

                    # print('in adv noise func')
                    if torch.sum(torch.isnan(adv_example_for_each_model[name]))>0:
                        print('in epsilon update',name,'input',torch.sum(torch.isnan(delta_x_clone)),'mask',torch.sum(torch.isnan(mask)),'deltax',torch.sum(torch.isnan(self.delta_x)))
                        exit()
                if 'mask' in inter_grad_target:
                    if 'mask' not in inter_grad:
                        inter_grad['attack']['mask']={}
                    mask_clone=mask.clone()
                    mask_clone.register_hook(print_grad(name,'mask'))
                    if isinstance(x,dict):
                        adv_example_for_each_model[name]=self.delta_x*mask_clone+x[name]
                    else:
                        adv_example_for_each_model[name]=self.delta_x*mask_clone+x
            else:
                if inter_grad!=None:
                    inter_grad['attack']['delta_x']={}
                if isinstance(x,dict):
                    adv_example_for_each_model[name]=self.delta_x+x[name]
                else:      
                    adv_example_for_each_model[name]=self.delta_x+x
                adv_example_for_each_model[name].register_hook(print_grad(name,'delta_x'))
            # adv_example_for_each_model[name]=self.delta_x+x
            
            # adv_example_for_each_model[name].register_hook(lambda grad: print('biggest grad for '+str(i)+' is ',torch.max(torch.abs(grad))))
            
            # print(index)
            # index+=1
        # exit()
        return adv_example_for_each_model
            # return self.delta_x+x
    def update_func(self):
        if self.a !=None and self.b!=None:
            self.a.data=torch.clamp(self.a,min=-1,max=1).detach()
            self.b.data=torch.clamp(self.b,min=-1,max=1).detach()
        
if __name__=='__main__':
    noise=Noise_Func((1,3,112,112),1,0)
    noise.to('cuda')
    print(noise.parameters())