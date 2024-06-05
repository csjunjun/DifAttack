import os
from whiteboxAttack import TIMIFGSM as timMultiModel
from whiteboxAttack import SINIFGSM as siniMultiModel
from whiteboxAttack import VNIFGSM as vniMultiModel
from config import refs,popdict
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from autoencoder import Autoencoder

from torchvision import transforms

from torchvision.datasets import ImageFolder

from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,LinfMomentumIterativeAttack
from advertorch.context import ctx_noparamgrad_and_eval
import plusUtils
import random

def setSeed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False       
    torch.backends.cudnn.deterministic = True

#image values are clamped in range of 0 & 1 to get rid of negative values 
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 224, 224)
    return x
  

def mysortkey(filename:str):
    return int(filename.split("_")[0]) 
      
def test(npop,targeted,target_label,modelchoice,attackType,\
             change=True,outputn="",saveres=False,scenario="close"):
    if not os.path.exists(outputn) and saveres:
        os.mkdir(outputn)
        print(f"mkdir {outputn}")
    adv_models= plusUtils.load_adv_imagenet(ref.split(",")) 

    sigma = 0.1
    sigma_f=0.1   
    lr=0.01 
    i = 0 
    linf_con = 0.05 
    
    print("targeted:{},target_label:{} sigma:{} sigma_f:{} lr:{} npop:{}".format(targeted,float(target_label),sigma,sigma_f,lr,npop))
    
    modelpathp=f'{targetmodename}.pth.tar'
    
    model_clean.load_state_dict(torch.load(modelpathp)['state_dict_clean'])
    model_adv.load_state_dict(torch.load(modelpathp)['state_dict_adv'])
    print(modelpathp)
    model_clean.eval()
    model_adv.eval()
    succ_list=[]
    query_list,l2_list,linf_list = [],[],[]

    imgbase = '/data/junliu/ImageNet_val_mini/randomCropped224'      
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
        if scenario=="close":
            batch_surro_adv,_ = augmentationTest(batch.cuda(),label.cuda(),\
                                                targeted=targeted,attackType=attackType,adv_models=adv_models,change=change,\
                                                    ref=ref)
            batch_surro_adv = (batch_surro_adv-0.5)/0.5
            start_point = batch_surro_adv
        batch = (batch-0.5)/0.5
        if scenario=="open":
            start_point = batch
        batch = batch.cuda()
        label = label.cuda()
        with torch.no_grad():
            output = net(batch*0.5+0.5)
        pre=torch.argmax(output,dim=1)
        if target_label>=0 and pre==target_label:
            continue
        elif pre != label:
            continue
        # ===================forward=====================
        with torch.no_grad():
            if modelchoice=="adv":
                
                output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_adv(start_point) 
            elif modelchoice=="clean":
                output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_clean(start_point) 

        mu = sigma*torch.randn_like(batch).detach().cuda()
        query=0
        succ = False

        while query<10000:
            mu_z = torch.randn((npop,3,224,224)).cuda()
            modify = mu.repeat(npop,1,1,1)+sigma_f*mu_z
            batch_perturb = start_point.repeat(npop,1,1,1)+ modify
            with torch.no_grad():
                output_p ,z_p_vis0,z_p_vis,z2_p_vis,z3_p_vis,z4_p_vis,\
                    z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem\
                        = model_adv(batch_perturb) 

            
            with torch.no_grad():
                output_inter1 = model_adv.decode(z_vis0.repeat(npop,1,1,1),z_vis.repeat(npop,1,1,1),\
                                                z2_vis.repeat(npop,1,1,1),z3_vis.repeat(npop,1,1,1),\
                                                z4_vis.repeat(npop,1,1,1),z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem)
            
            if linf_con>0:
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
            

            # print('Img:{} query:{} advL2:{:.6f} advLinf:{:.6f} , minlossblack:{:.6f}\n'\
            #     .format(str(i+1),query, torch.mean(advl2).data,torch.mean(advlinf).data,\
            #             torch.min(loss_black).data))
            
            succ_res = torch.where(succ==True)
            if len(succ_res[0])>0:
                succ_l2 = advl2[succ_res]
                succ_linf = advlinf[succ_res]
                minidx = torch.argmin(succ_l2)
                succ_list.append(i)
                query_list.append(query)
                l2_list.append(float(succ_l2[minidx]))
                linf_list.append(float(succ_linf[minidx]))
                if saveres:
                
                    Image.fromarray(np.array(np.round((output_inter1[minidx]*0.5+0.5).permute(1,2,0).detach().cpu().numpy()*255),dtype=np.uint8)).save(f"{outputn}/{i}_{int(label)}.png")
                    np.save(f"{outputn}/Resnet18_{i}_adv_target-1_label{int(label)}.npy",(output_inter1[minidx]*0.5+0.5).detach().cpu().numpy())

                del advl2,advlinf
                break
            else:
                Reward = -loss_black
                A      = (Reward - torch.mean(Reward))/(torch.std(Reward) + 1e-10)
                mu    += (lr/ (npop * sigma_f))*(torch.matmul(mu_z.flatten(start_dim=1).t(), A.view(-1, 1))).view(1, -1).reshape(-1,3,224,224)
                del A
        
        i+= len(batch)
        # if (i+1)%10==0:
        #     print("Succ rate:{:.4f}".format(len(succ_list)/i))
        #     print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
        #     print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
        #     print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))    

        #     print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
        #     print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
        #     print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))  
        torch.cuda.empty_cache()      
        if i>=1000:
            break
    print("Succ rate:{:.4f}".format(len(succ_list)/i))
    print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
    print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
    print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))

    print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
    print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
    print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))

    return len(succ_list)/i,np.mean(np.asarray(query_list)),np.median(np.asarray(query_list))

def augmentation(x, true_lab, targeted=False,attackType="pgd"):

    model_idx = np.random.randint(0, len(adv_models))
    model_chosen = adv_models[model_idx]
    if attackType =="pgd":
        adversary = LinfPGDAttack(
            model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
            nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=targeted)
    elif attackType =="cw":
        adversary = CarliniWagnerL2Attack(model_chosen,1000,confidence=0,targeted=targeted,learning_rate=0.2,binary_search_steps=1
                                          ,max_iterations=10000) 
    elif attackType =="mifgsm":
        adversary = LinfMomentumIterativeAttack(model_chosen,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.05,targeted=targeted) 
        
    elif attackType =="multi2":
        a = random.randint(0, 1)
        if a ==0:
            adversary = LinfPGDAttack(
                model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
                nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=targeted) 
        else:
            adversary = LinfMomentumIterativeAttack(model_chosen,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.05,targeted=targeted) 
                      
        
    elif attackType =="multi":
        

        a = random.randint(0, 1)
        if a ==0:
            adversary = LinfPGDAttack(
                model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
                nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=targeted) 
        else:
            adversary = CarliniWagnerL2Attack(model_chosen,1000,confidence=0,targeted=targeted,learning_rate=0.2,binary_search_steps=1
                                            ,max_iterations=10000)    
    elif attackType =="multi3":
        

        a = random.randint(0, 2)
        if a ==0:
            adversary = LinfPGDAttack(
                model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
                nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=targeted) 
        elif a==1:
            adversary = CarliniWagnerL2Attack(model_chosen,1000,confidence=0,targeted=targeted,learning_rate=0.2,binary_search_steps=1
                                            ,max_iterations=10000)   
        else:
            adversary = LinfMomentumIterativeAttack(model_chosen,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.05,targeted=targeted) 

                
    with ctx_noparamgrad_and_eval(model_chosen):
        xadv = adversary.perturb(x, true_lab if targeted else None)
        adv_label = adversary._get_predicted_label(xadv)
        #print("xadv norm:{}".format(torch.mean(torch.norm(xadv.flatten(start_dim=1)-x.flatten(start_dim=1),dim=0))))
        #save_image(xadv,'test.png')
        if targeted:
            attack_acc= len(torch.where(adv_label==true_lab)[0])
        else:
            attack_acc= len(torch.where(adv_label!=true_lab)[0])
        #print("augmentation attack acc:{:.4f}".format(attack_acc/len(adv_label)))
        return xadv,adv_label
    
    
def augmentationTest(x, true_lab, targeted=False,attackType="pgd",\
                     adv_models=None,change=True,ref=""):

    if attackType=="timifgsm":
        adversary = timMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=30,random_start=True,targeted=targeted,
            change=change
        )  
    elif attackType=="sinifgsm":
        adversary = siniMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted
        )  
    elif attackType=="sinitifgsm":
        step=10
        if targeted and (ref.find("SwinV2T")>-1 or ref.find("ConvNextBase")>-1\
                         or ref.find("ResNet101")>-1 or ref.find("EfficientB3")>-1):
            step=30
        adversary = siniMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=step,targeted=targeted,tidi=True
        ) 
    elif attackType=="vnifgsm":
        adversary = vniMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted
        )
    elif attackType=="vnitifgsm":
        adversary = vniMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted,tidi=True
        )   
    else:
        raise NotImplementedError      

    xadv = adversary(x,true_lab)
    return xadv,None
           

def adv_loss( y, label,target=-1,models=None):
    loss = 0.
    margin=innermargin
    if models is None:
        models = adv_models
    for adv_model in models:
        with torch.no_grad():
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
            #margin = diff
        loss += margin.mean()
    loss /= len(models)
        
    return loss 

def drawLoss(avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5):
    x = [x for x in range(len(avg_loss))]
    plt.figure()
    plt.plot(x,avg_loss)
    plt.savefig("{}/avg_loss.jpg".format(outputname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss1)
    plt.savefig("{}/avg_loss1.jpg".format(outputname))
    plt.close()
    plt.figure()
    plt.plot(x,avg_loss3)
    plt.savefig("{}/avg_loss3.jpg".format(outputname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss4)
    plt.savefig("{}/avg_loss4.jpg".format(outputname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss5)
    plt.savefig("{}/avg_loss5.jpg".format(outputname))
    plt.close()

def adv_loss_train( y, label,target=False,randommargin=False):
    loss = 0.
    
    for adv_model in adv_models:
        logits = adv_model(y)
        if randommargin:
            randommargin = int(torch.randint(0,innermargin,(1,)))
        if not target:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0]
            if not randommargin:
                margin = torch.nn.functional.relu(diff + innermargin, True) - innermargin
            else:
                margin = torch.nn.functional.relu(diff + randommargin, True) - randommargin
        else:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0] - logits[one_hot]
            if not randommargin:
                margin = torch.nn.functional.relu(diff + innermargin, True) - innermargin
            else:
                margin = torch.nn.functional.relu(diff + randommargin, True) - randommargin
        loss += margin.mean()
    loss /= len(adv_models)
        
    return loss 
    
def train():
    MSE = nn.MSELoss()
    attackType = "pgd"  
    
    rm = False
    print("attackType={},randommargin={}".format(attackType,rm)) 

    total_iter = 0
    avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5=[],[],[],[],[]
    for epoch in range(num_epochs):
        total_loss,total_loss1,total_loss2,total_loss3,total_loss4,total_loss5 = 0,0,0,0,0,0
        totalimg = 0
        innerloop=0
        for _,data in enumerate(train_loader):
            model_clean.train()
            model_adv.train()
            batch, label = data       # Get a batch,-1,1
            batch = (batch).cuda()
            label = label.cuda()
            batch_adv,adv_label = augmentation(batch*0.5+0.5,label,targeted=False,attackType=attackType)
            batch_adv = (batch_adv-0.5)/0.5
            

            output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_clean(batch)   
            output_adv ,z_adv_vis0,z_adv_vis,z2_adv_vis,z3_adv_vis,\
                z4_adv_vis,z_adv_sem0,z_adv_sem,z2_adv_sem,z3_adv_sem,z4_adv_sem= model_adv(batch_adv)      
            totalimg += len(batch)
            #MSE 
            
            output_inter1 = model_adv.decode(z_vis0,z_vis,z2_vis,z3_vis,z4_vis,z_adv_sem0,z_adv_sem,z2_adv_sem,z3_adv_sem,z4_adv_sem)
            output_inter2 = model_clean.decode(z_adv_vis0,z_adv_vis,z2_adv_vis,z3_adv_vis,z4_adv_vis,z_sem0,z_sem,z2_sem,z3_sem,z4_sem)
            
            loss1= MSE(output,batch)+MSE(output_adv,batch_adv)
            loss3 = MSE(output_inter1,batch)+MSE(output_inter2,batch_adv)
            
            loss4 = adv_loss_train(output_inter1*0.5+0.5,label,randommargin=rm) 
            loss5 = adv_loss_train(output_inter2*0.5+0.5,label,target=True,randommargin=rm)

            loss =loss1+loss3+loss4+loss5
            

            # ===================backward====================
            optimizer_clean.zero_grad()
            optimizer_adv.zero_grad()
            loss.backward()
            optimizer_clean.step()
            optimizer_adv.step()


            # ===================log========================
            total_loss += loss.data
            total_loss1 += loss1.data
            total_loss3 += loss3.data
            total_loss4 += loss4.data
            total_loss5 += loss5.data

            if innerloop%100 ==0:
                print('\nepoch [{}/{}], loss:{:.6f}, loss1:{:.6f}, loss3:{:.6f}, loss4:{:.6f}, loss5:{:.6f}\n'
                .format(epoch+1, num_epochs, total_loss/totalimg,total_loss1/totalimg,\
                                    total_loss3/totalimg,total_loss4/totalimg,total_loss5/totalimg))
            innerloop+=1
            total_iter+=1
            if total_iter%30==0:
                avg_loss.append(float(total_loss/totalimg))
                avg_loss1.append(float(total_loss1/totalimg))
                avg_loss3.append(float(total_loss3/totalimg))
                avg_loss4.append(float(total_loss4/totalimg))
                avg_loss5.append(float(total_loss5/totalimg))

                drawLoss(avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5)

            if innerloop%1000==0:
                out = torch.cat((batch[:2],output[:2]),0)
                out = to_img(out.cpu().data)
                #save output and input images
                #save_image(out, '{}/imagerecWeight_{}_{}.png'.format(outputname,epoch+1,innerloop))
            if innerloop%1000==0:
                save_dict = {
                'epoch': epoch + 1,
                'state_dict_clean': model_clean.state_dict(),
                'state_dict_adv': model_adv.state_dict(),
              
            }
                
                torch.save(save_dict, '{}/AE_model_{}_{}.pth.tar'.format(outputname,epoch+1,innerloop))

                save_dict = {
                'epoch': epoch + 1,
                'optimizer_clean' : optimizer_clean.state_dict(),
                'optimizer_adv' : optimizer_adv.state_dict()    
            }
                
                torch.save(save_dict, '{}/AE_model_{}_{}'.format(outputname,epoch+1))

        save_dict = {
        'epoch': epoch + 1,
        'state_dict_clean': model_clean.state_dict(),
        'state_dict_adv': model_adv.state_dict(),
                } 
        torch.save(save_dict, '{}/AE_model_{}_{}'.format(outputname,epoch+1,innerloop))

        save_dict = {
        'epoch': epoch + 1,
       
                'optimizer_clean' : optimizer_clean.state_dict(),
                'optimizer_adv' : optimizer_adv.state_dict() 
                } 
        torch.save(save_dict, '{}/AE_model_{}_{}'.format(outputname,epoch+1))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    setSeed(0)
    outputname='/data/junliu/weights/difAttackPlus'
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    torch.set_num_threads(1)
    para_dict={
    'num_epochs' :1,
    'batch_size' :16,
    'learning_rate' : 1e-4,
    'innermargin' :5,
    'resume':False,
    'resumePath':"GoogleNet.pth.tar",
    'resumePathOpt':"GoogleNet.pth.tar"

    }


    print("para_dict for training ImageNet:{}".format(para_dict))
    num_epochs = para_dict['num_epochs']
    batch_size = para_dict['batch_size']
    learning_rate = para_dict['learning_rate']
    innermargin = para_dict['innermargin']

    model_clean = Autoencoder().cuda()
    model_adv = Autoencoder().cuda()


    
    targetmodename='VGG16'#VGG16,SqueezeNet,GoogleNet,ResNet18
    ##targetmodename='SwinV2T'ConvNextBase,EfficientB3,SwinV2T,ResNet101
    if targetmodename in refs.keys():
        ref = refs[targetmodename]
        print(ref)
    else:
        raise NotImplementedError
    mode="test"
    if mode=="train":
        #Optimizer
        optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)
        optimizer_adv = torch.optim.Adam(model_adv.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)

        adv_models= plusUtils.load_adv_imagenet(ref.split(",")) 
        img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5), (0.5))
        ])

        imagenet_traindata = ImageFolder('/data/junliu/imgnet/tfds/manual/train',transform=img_transform)
        train_loader = torch.utils.data.DataLoader(imagenet_traindata,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)

        if not os.path.exists(outputname):
            os.mkdir(outputname)


        if para_dict['resume']:
            model_clean.load_state_dict(torch.load(para_dict['resumePath'])['state_dict_clean'])
            model_adv.load_state_dict(torch.load(para_dict['resumePath'])['state_dict_adv'])

            optimizer_clean.load_state_dict(torch.load(para_dict['resumePathOpt'])['optimizer_clean'])
            optimizer_adv.load_state_dict(torch.load(para_dict['resumePathOpt'])['optimizer_adv'])

        train()
        
    else:

        targeted=False
        target_label = -1 
        #target_label =torch.tensor([864]).cuda()
        scenario = "close"
        attackType = "sinitifgsm" #set as sinitifgsm in close-set scenarios;
        modelchoice="adv" #set adv/clean as in close-set/open-set scenarios;
        if targetmodename in popdict.keys():
            npop = popdict[targetmodename][1 if targeted else 0]
        else:
            raise NotImplementedError
        net = plusUtils.load_adv_imagenet([targetmodename])[0] #target model
        
        change=True
        saveres=False
        useclip = False
        saveoutputn = f"advimgs/{targetmodename}_{modelchoice}_{attackType}"
    
        print(f"modelchoice:{modelchoice}")
        print(f"attackType:{attackType}")
        print(f"useclip:{useclip}")
        print(f"change:{change}")
        print(f"save:{saveres} saveoutputn:{saveoutputn}")
                    
        asri,q,_=test(npop=npop,targeted=targeted,target_label=target_label,\
                      modelchoice=modelchoice,attackType=attackType,change=change,\
                        outputn=saveoutputn,saveres=saveres,scenario=scenario)
             