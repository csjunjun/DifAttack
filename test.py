
#imports
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch

import numpy as np
from torch import nn
from torchvision import transforms
import utils
from autoencoder import Autoencoder

seed = 0
def setSeed(seed):
    np.random.seed(seed) #[0,2^32-1]
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False       
    torch.backends.cudnn.deterministic = True
setSeed(0)



def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 224, 224)
    return x

   
    

def mysortkey(filename:str):
    return int(filename.split("_")[0]) 

def test(model):
    with torch.no_grad():
        MSE = nn.MSELoss()  #define mean square error loss
        sigma = 0.1
        sigma_f=0.1 
       
        lr=0.01 
        i = 0
        linf_con = 0.05 
        
        #target_label=torch.tensor([864]).cuda()
        #target_label=torch.tensor([776]).cuda()
        target_label=-1
        if target_label==-1:
            npop=8
        else:
            npop=12      #slight change the npop if the performance is not satisfactory. 
        print("target_label:{} sigma:{} sigma_f:{} lr:{} npop:{} ".format(float(target_label),sigma,sigma_f,lr,npop))
        if target_netname=='Resnet18':
            modelp = 'resnet18.pth.tar'
            
        elif target_netname=='Squeezenet':
           
            modelp='squeezenet.pth.tar'
        elif target_netname=='VGG16':
            
            modelp = 'vgg.pth.tar'    
            
        elif target_netname=='Googlenet':
            modelp = 'google.pth.tar'
           
        print("loading model :{}".format(modelp))
        model.load_state_dict(torch.load(modelp)['state_dict'])

        model.eval()
        succ_list=[]
        fail_list,queries = [],[]
        
        imgbase = 'randomCropped224'
        imagelist = [f for f in os.listdir(imgbase)]
        imagelist.sort(key=mysortkey)
        from PIL import Image
        trans = transforms.Compose([
                transforms.ToTensor()
                ])
       
        for filename in imagelist:

            
            imgpath = "{}/{}".format(imgbase,filename)
            label = torch.tensor([int(filename.split("_")[-1].split(".")[0])])

            batch = Image.open(imgpath)
            batch = batch.convert("RGB")
            batch = trans(batch).unsqueeze(0)
            

            lower = torch.clamp(batch-linf_con,0,1).cuda()
            upper = torch.clamp(batch+linf_con,0,1).cuda()
            batch = (batch-0.5)/0.5
            
           
            batch = batch.cuda()
            label = label.cuda()
            with torch.no_grad():
                output = net(batch*0.5+0.5)
            pre=torch.argmax(output,dim=1)
            if pre != label:
                continue
            if target_label>=0 and pre==target_label:
                continue
           
            
            # ===================forward=====================
            with torch.no_grad():
                output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model(batch) 
            print("rec l2:{}".format(float(torch.norm(output*0.5-batch*0.5))))
            
            mu = sigma*torch.randn_like(batch).detach().cuda()
            query=0
            succ = False

            while query<10000:
           
                mu_z = torch.randn((npop,3,224,224)).cuda()
                modify = mu.repeat(npop,1,1,1)+sigma_f*mu_z
                batch_perturb = batch.repeat(npop,1,1,1)+ modify
                with torch.no_grad():
                    output_p ,z_p_vis0,z_p_vis,z2_p_vis,z3_p_vis,z4_p_vis,z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem= model(batch_perturb) 
      
                with torch.no_grad():
                    output_inter1 = model.decode(z_vis0.repeat(npop,1,1,1),z_vis.repeat(npop,1,1,1),\
                                                 z2_vis.repeat(npop,1,1,1),z3_vis.repeat(npop,1,1,1),\
                                                    z4_vis.repeat(npop,1,1,1),z_p_sem0,z_p_sem,\
                                                        z2_p_sem,z3_p_sem,z4_p_sem)
                    
                loss1= MSE(output,batch)
                loss3 = MSE(output_inter1,batch)
               
                        
               
                output_inter1 = (torch.clamp(output_inter1*0.5+0.5,lower,upper)-0.5)/0.5
                with torch.no_grad():
                    adv_logits = net(output_inter1*0.5+0.5)
               
                adv_pre=torch.argmax(adv_logits,dim=1)
                del adv_logits
                if target_label>=0:
                    succ = adv_pre==target_label
                else:
                    succ = adv_pre!=label
                query+=npop

                
                loss_black = None
                for jj in range(npop):
                    if loss_black is None:
                        loss_black=adv_loss(output_inter1[jj].unsqueeze(0)*0.5+0.5,label,target=target_label,models=[net]).unsqueeze(0)
                    else:
                        loss_black = torch.cat((loss_black,adv_loss(output_inter1[jj].unsqueeze(0)*0.5+0.5,label,target=target_label,models=[net]).unsqueeze(0)),dim=0)


                advl2 = torch.norm((output_inter1*0.5-batch*0.5).flatten(start_dim=1),dim=1)
                advlinf = torch.norm((output_inter1*0.5-batch*0.5).flatten(start_dim=1),dim=1,p=np.inf)
               
                if query%(npop*5) == 0:
                    print('Img:{} query:{} advL2:{:.6f} advLinf:{:.6f} loss1:{:.6f},  loss3:{:.6f}, minlossblack:{:.6f}\n'\
                        .format(str(i+1),query, torch.mean(advl2).data,torch.mean(advlinf).data,loss1.data,loss3.data,\
                                torch.min(loss_black).data))
                
                succ_res = torch.where(succ==True)
                if len(succ_res[0])>0:
                
                    succ_list.append(i)  
                    break
                else:
                    Reward = -loss_black
                    if npop  >1:

                        A      = (Reward - torch.mean(Reward))/(torch.std(Reward) + 1e-10)
                    else:
                        A = Reward
                    mu    += (lr/ (npop * sigma_f))*(torch.matmul(mu_z.flatten(start_dim=1).t(), A.view(-1, 1))).view(1, -1).reshape(-1,3,224,224)
                    del A
           
            
            fail_list.append(i)  
            queries.append(query)
            
            i+= len(batch)
             
            torch.cuda.empty_cache()      
            if i>=1000:
                break
            if i%50==0:
                print("Succ rate:{:.4f}".format(len(succ_list)/i))
                print("Avg.query:{:.4f}".format(np.mean(np.asarray(queries))))
                

                
        print("Succ rate:{:.4f}".format(len(succ_list)/i))
        print("Avg.query:{:.4f}".format(np.mean(np.asarray(queries))))
       


def adv_loss( y, label,target=-1,models=None):
    loss = 0.
    margin=0

    for adv_model in models:

        logits = adv_model(y)

        if target==-1:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0]
            margin = torch.nn.functional.relu(diff + margin, True) - margin
        else:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = target.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0] - logits[one_hot]
            margin = torch.nn.functional.relu(diff + margin, True) - margin
           
        loss += margin.mean()
    loss /= len(models)
        
    return loss 

target_netname='VGG16'


net = utils.load_adv_imagenet([target_netname])[0] #target model

print("target modeL:{} ".format(target_netname))

model = Autoencoder().cuda()
       
test(model)

