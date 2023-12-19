import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from advertorch.attacks import LinfPGDAttack,LinfMomentumIterativeAttack
from advertorch.context import ctx_noparamgrad_and_eval
import utils
from autoencoder import Autoencoder
from PIL import Image
import random
def setSeed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False       
    torch.backends.cudnn.deterministic = True

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 224, 224)
    return x


def mysortkey(filename:str):
    return int(filename.split("_")[0]) 
     

def augmentation(x, true_lab, targeted=False):

    model_idx = np.random.randint(0, len(adv_models))
    model_chosen = adv_models[model_idx]
    #PGD
    adversary = LinfPGDAttack(
        model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
        nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=targeted)
    #MIFGSM
    #adversary = LinfMomentumIterativeAttack(model_chosen,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.05,targeted=targeted) 

    #MIFGSM or PGD
    # a = random.randint(0, 1)
    # if a ==0:
    #     adversary = LinfPGDAttack(
    #         model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
    #         nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
    #         clip_max=1.0, targeted=targeted) 
    # else:
    #     adversary = LinfMomentumIterativeAttack(model_chosen,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.05,targeted=targeted) 


    with ctx_noparamgrad_and_eval(model_chosen):
        x_adv = adversary.perturb(x, true_lab if targeted else None)
        adv_label = adversary._get_predicted_label(x_adv)
        # if targeted:
        #     attack_acc= len(torch.where(adv_label==true_lab)[0])
        # else:
        #     attack_acc= len(torch.where(adv_label!=true_lab)[0])
        # print("augmentation attack acc:{:.4f}".format(attack_acc/len(adv_label)))
    return x_adv,adv_label




def adv_loss_train( y, label,target=False):
    loss = 0.
    margin=5
    for adv_model in adv_models:
        logits = adv_model(y)

        if not target:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0]
            margin = torch.nn.functional.relu(diff + margin, True) - margin
        else:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0] - logits[one_hot]
            margin = torch.nn.functional.relu(diff + margin, True) - margin
        loss += margin.mean()
    loss /= len(adv_models)
        
    return loss 

def drawLoss(avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5,outputname):
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


    
def train(para_dict,model):
    num_epochs = para_dict['num_epochs']
    learning_rate = para_dict['learning_rate']
    outputname = para_dict['outputname']

    MSE = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
    #Training
    total_iter = 0
    avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5=[],[],[],[],[]
    for epoch in range(num_epochs):
        total_loss,total_loss1,total_loss3,total_loss4,total_loss5 = 0,0,0,0,0
        totalimg = 0
        innerloop=0
        for _,data in enumerate(train_loader):
            model.train()
            batch, label = data       # Get a batch,-1,1
            batch = (batch).cuda()
            label = label.cuda()
            batch_adv,adv_label = augmentation(batch*0.5+0.5,label,targeted=False)
            batch_adv = (batch_adv-0.5)/0.5
            # ===================forward=====================

            output,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,z_adv0,z_adv,z2_adv,z3_adv,z4_adv= model(batch) 
            output_adv ,z_adv_vis0,z_adv_vis,z2_adv_vis,z3_adv_vis,z4_adv_vis,z_adv_adv0,z_adv_adv,z2_adv_adv,z3_adv_adv,z4_adv_adv= model(batch_adv)      
            totalimg += len(batch)
            #MSE 
            
            output_inter1 = model.decode(z_vis0,z_vis,z2_vis,z3_vis,z4_vis,z_adv_adv0,z_adv_adv,z2_adv_adv,z3_adv_adv,z4_adv_adv)
            output_inter2 = model.decode(z_adv_vis0,z_adv_vis,z2_adv_vis,z3_adv_vis,z4_adv_vis,z_adv0,z_adv,z2_adv,z3_adv,z4_adv)
            
            loss1= MSE(output,batch)+MSE(output_adv,batch_adv)
            loss3 = MSE(output_inter1,batch)+MSE(output_inter2,batch_adv)

            loss4 = adv_loss_train(output_inter1*0.5+0.5,label) 
            loss5 = adv_loss_train(output_inter2*0.5+0.5,label,target=True)

            loss =loss1+loss3+loss4+loss5

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================log========================
            total_loss += loss.data
            total_loss1 += loss1.data
            total_loss3 += loss3.data
            total_loss4 += loss4.data
            total_loss5 += loss5.data

            if innerloop%100 ==0:
                print('\nepoch [{}/{}], loss:{:.6f}, loss1:{:.6f}, loss3:{:.6f}, loss4:{:.6f}, loss5:{:.6f}\n'
                .format(epoch, num_epochs, total_loss/totalimg,total_loss1/totalimg,\
                                    total_loss3/totalimg,total_loss4/totalimg,total_loss5/totalimg))
            innerloop+=1
            total_iter+=1
            if total_iter%30==0:
                avg_loss.append(float(total_loss/totalimg))
                avg_loss1.append(float(total_loss1/totalimg))
                avg_loss3.append(float(total_loss3/totalimg))
                avg_loss4.append(float(total_loss4/totalimg))
                avg_loss5.append(float(total_loss5/totalimg))

                drawLoss(avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5,outputname)

            if innerloop%1000==0:
                out = torch.cat((batch[:2],output[:2]),0)
                out = to_img(out.cpu().data)
               
                save_image(out, '{}/imagerec_{}_{}.png'.format(outputname,epoch,innerloop))
            if innerloop%2000==0:
                save_dict = {
                'epoch': epoch ,
                'state_dict': model.state_dict(),
            }
                torch.save(save_dict, '{}/AE_imagenet_{}_{}.pth.tar'.format(outputname,epoch,innerloop))

    
        save_dict = {
        'epoch': epoch ,
        'state_dict': model.state_dict(),
    }
        torch.save(save_dict, '{}/AE_imagenet_{}_{}.pth.tar'.format(outputname,epoch,innerloop))


if __name__ == "__main__":
    #Hyperparameters
    para_dict={
    'seed' : 0,
    'stage':"train",
    'num_epochs' :2, #2 for imagenet, 300 for cifar-10
    'batch_size' :32,
    'learning_rate' : 1e-3,
    'outputname': './exp',
    
    'sigma':0.1,
    'sigma_f':0.1 ,
    'lr':0.01 ,
    'linf_con' :0.05,
    'targeted':False,
    'target_label':torch.tensor([864]).cuda(),
    #target_label=torch.tensor([776]).cuda()
    'target_netname':'Resnet18',
    #'modelp':'./PretrainModels/ckpt.pth.tar'
    'modelp':'/data/junliu/weights/deCouplingAttack/PretrainModelsImageNetMultiConv_resume/AE_model_imagerecWeight_2_45400.pth.tar'
    }
    print("para_dict for training ImageNet:{}".format(para_dict))

    setSeed(para_dict["seed"])
    
    model = Autoencoder().cuda()
    if para_dict["stage"] == "train":
        ref="Resnet18,Googlenet,Squeezenet" #VGG16
        print(ref)
        adv_models= utils.load_adv_imagenet(ref.split(","))  
        
        img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5), (0.5))
        ])

        imagenet_traindata = ImageFolder('/data/ILSVRC2012/train',transform=img_transform)

        train_loader = torch.utils.data.DataLoader(imagenet_traindata,
                                                batch_size=para_dict["batch_size"],
                                                shuffle=True,
                                                num_workers=4)
        
        #Directory for saving intermediate images & models
        if not os.path.exists('{}'.format(para_dict["outputname"])):
            os.mkdir('{}'.format(para_dict["outputname"]))

        train(para_dict)
    
   