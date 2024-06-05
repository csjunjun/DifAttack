from torch import nn
import torch.nn.functional as F
import torch

class Autoencoder(nn.Module):
    def __init__(self,before_combine_weight=False):
        super(Autoencoder,self).__init__()
        #Encoder
        self.conv_1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)

        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)

        self.conv_5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(512)

        self.conv_6 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(512)

        #Decoder
        self.deconv_0 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=0)
        self.batchNorm0_d = nn.BatchNorm2d(256)

        self.deconv_1 = nn.ConvTranspose2d(512+256, 256, 5, stride=4, padding=1, output_padding=1)
        self.batchNorm1_d = nn.BatchNorm2d(256)

        self.deconv_2 = nn.ConvTranspose2d(256+128, 128, 3, stride=2, padding=1, output_padding=1)
        self.batchNorm2_d = nn.BatchNorm2d(128)
        
        self.deconv_3 = nn.ConvTranspose2d(128+64, 64, 3, stride=2, padding=1, output_padding=1)
        self.batchNorm3_d = nn.BatchNorm2d(64)
        
        self.deconv_4 = nn.ConvTranspose2d(64+32, 32, 3, stride=2, padding=1, output_padding=1)
        self.batchNorm4_d = nn.BatchNorm2d(32)
        
        self.deconv_5 = nn.ConvTranspose2d(32, 3, 3,  padding=1)

        self.weight0_vis = nn.Conv2d(512,512,1,1)
        self.weight1_vis = nn.Conv2d(512,512,1,1)
        self.weight2_vis = nn.Conv2d(128,128,1,1)
        self.weight3_vis = nn.Conv2d(64,64,1,1)
        self.weight4_vis = nn.Conv2d(32,32,1,1)

        self.weight0_sem = nn.Conv2d(512,512,1,1)
        self.weight1_sem = nn.Conv2d(512,512,1,1)
        self.weight2_sem = nn.Conv2d(128,128,1,1)
        self.weight3_sem = nn.Conv2d(64,64,1,1)
        self.weight4_sem = nn.Conv2d(32,32,1,1)

        self.weight0_vis_vis = nn.Conv2d(512,512,1,1)
        self.weight1_vis_vis = nn.Conv2d(512,512,1,1)
        self.weight2_vis_vis = nn.Conv2d(128,128,1,1)
        self.weight3_vis_vis = nn.Conv2d(64,64,1,1)
        self.weight4_vis_vis = nn.Conv2d(32,32,1,1)

        self.weight0_vis_adv = nn.Conv2d(512,512,1,1)
        self.weight1_vis_adv = nn.Conv2d(512,512,1,1)
        self.weight2_vis_adv = nn.Conv2d(128,128,1,1)
        self.weight3_vis_adv = nn.Conv2d(64,64,1,1)
        self.weight4_vis_adv = nn.Conv2d(32,32,1,1)

        self.weight0_sem_vis = nn.Conv2d(512,512,1,1)
        self.weight1_sem_vis = nn.Conv2d(512,512,1,1)
        self.weight2_sem_vis = nn.Conv2d(128,128,1,1)
        self.weight3_sem_vis = nn.Conv2d(64,64,1,1)
        self.weight4_sem_vis = nn.Conv2d(32,32,1,1)

        self.weight0_sem_adv = nn.Conv2d(512,512,1,1)
        self.weight1_sem_adv = nn.Conv2d(512,512,1,1)
        self.weight2_sem_adv = nn.Conv2d(128,128,1,1)
        self.weight3_sem_adv = nn.Conv2d(64,64,1,1)
        self.weight4_sem_adv = nn.Conv2d(32,32,1,1)


        self.combine0_adv = nn.Conv2d(512*2,512,1,1)
        self.combine1_adv = nn.Conv2d(512*2,512,1,1)
        self.combine2_adv = nn.Conv2d(128*2,128,1,1)
        self.combine3_adv = nn.Conv2d(64*2,64,1,1)
        self.combine4_adv = nn.Conv2d(32*2,32,1,1)

        self.combine0_vis = nn.Conv2d(512*2,512,1,1)
        self.combine1_vis = nn.Conv2d(512*2,512,1,1)
        self.combine2_vis = nn.Conv2d(128*2,128,1,1)
        self.combine3_vis = nn.Conv2d(64*2,64,1,1)
        self.combine4_vis = nn.Conv2d(32*2,32,1,1)

        self.combine0 = nn.Conv2d(512*2,512,1,1)
        self.combine1 = nn.Conv2d(512*2,512,1,1)
        self.combine2 = nn.Conv2d(128*2,128,1,1)
        self.combine3 = nn.Conv2d(64*2,64,1,1)
        self.combine4 = nn.Conv2d(32*2,32,1,1)

        self.before_combine_weight = before_combine_weight


    def forward(self, x):
       # Encoder
        conv_b1 = F.relu(self.batchNorm1(self.conv_1(x)))
        conv_b2 = F.relu(self.batchNorm2(self.conv_2(conv_b1)))
        conv_b3 = F.relu(self.batchNorm3(self.conv_3(conv_b2)))
        conv_b4 = F.relu(self.batchNorm4(self.conv_4(conv_b3)))
        conv_b5 = F.relu(self.batchNorm5(self.conv_5(conv_b4)))
        conv_b6 = F.relu(self.batchNorm6(self.conv_6(conv_b5)))

        #Decoupling 
        conv_b6_vis = self.weight0_vis(conv_b6) #512,4,4
        conv_b5_vis = self.weight1_vis(conv_b5) #512,7,7
        conv_b3_vis = self.weight2_vis(conv_b3) #128,28,28
        conv_b2_vis = self.weight3_vis(conv_b2) #64,56,56
        conv_b1_vis = self.weight4_vis(conv_b1) #32,112,112

        conv_b6_sem = self.weight0_sem(conv_b6)
        conv_b5_sem = self.weight1_sem(conv_b5)
        conv_b3_sem = self.weight2_sem(conv_b3)
        conv_b2_sem = self.weight3_sem(conv_b2)
        conv_b1_sem = self.weight4_sem(conv_b1)

        conv_b6_vis_vis = self.weight0_vis_vis(conv_b6_vis)
        conv_b5_vis_vis = self.weight1_vis_vis(conv_b5_vis)
        conv_b3_vis_vis = self.weight2_vis_vis(conv_b3_vis)
        conv_b2_vis_vis = self.weight3_vis_vis(conv_b2_vis)
        conv_b1_vis_vis = self.weight4_vis_vis(conv_b1_vis)

        conv_b6_vis_adv = self.weight0_vis_adv(conv_b6_vis)
        conv_b5_vis_adv = self.weight1_vis_adv(conv_b5_vis)
        conv_b3_vis_adv = self.weight2_vis_adv(conv_b3_vis)
        conv_b2_vis_adv = self.weight3_vis_adv(conv_b2_vis)
        conv_b1_vis_adv = self.weight4_vis_adv(conv_b1_vis)

        conv_b6_sem_vis = self.weight0_sem_vis(conv_b6_sem)
        conv_b5_sem_vis = self.weight1_sem_vis(conv_b5_sem)
        conv_b3_sem_vis = self.weight2_sem_vis(conv_b3_sem)
        conv_b2_sem_vis = self.weight3_sem_vis(conv_b2_sem)
        conv_b1_sem_vis = self.weight4_sem_vis(conv_b1_sem)

        conv_b6_sem_adv = self.weight0_sem_adv(conv_b6_sem)
        conv_b5_sem_adv = self.weight1_sem_adv(conv_b5_sem)
        conv_b3_sem_adv = self.weight2_sem_adv(conv_b3_sem)
        conv_b2_sem_adv = self.weight3_sem_adv(conv_b2_sem)
        conv_b1_sem_adv = self.weight4_sem_adv(conv_b1_sem)



       #Fuse        
        conv_b6_vis = self.combine0_vis(torch.cat((conv_b6_vis_vis,conv_b6_sem_vis),dim=1))
        conv_b5_vis = self.combine1_vis(torch.cat((conv_b5_vis_vis,conv_b5_sem_vis),dim=1))
        conv_b3_vis = self.combine2_vis(torch.cat((conv_b3_vis_vis,conv_b3_sem_vis),dim=1))
        conv_b2_vis = self.combine3_vis(torch.cat((conv_b2_vis_vis,conv_b2_sem_vis),dim=1))
        conv_b1_vis = self.combine4_vis(torch.cat((conv_b1_vis_vis,conv_b1_sem_vis),dim=1))

        conv_b6_sem = self.combine0_adv(torch.cat((conv_b6_vis_adv,conv_b6_sem_adv),dim=1))
        conv_b5_sem = self.combine1_adv(torch.cat((conv_b5_vis_adv,conv_b5_sem_adv),dim=1))
        conv_b3_sem = self.combine2_adv(torch.cat((conv_b3_vis_adv,conv_b3_sem_adv),dim=1))
        conv_b2_sem = self.combine3_adv(torch.cat((conv_b2_vis_adv,conv_b2_sem_adv),dim=1))
        conv_b1_sem = self.combine4_adv(torch.cat((conv_b1_vis_adv,conv_b1_sem_adv),dim=1))

        conv_b6 =  self.combine0(torch.cat((conv_b6_vis,conv_b6_sem),dim=1))
        conv_b5 =  self.combine1(torch.cat((conv_b5_vis,conv_b5_sem),dim=1))
        conv_b3 =  self.combine2(torch.cat((conv_b3_vis,conv_b3_sem),dim=1))
        conv_b2 =  self.combine3(torch.cat((conv_b2_vis,conv_b2_sem),dim=1))
        conv_b1 =  self.combine4(torch.cat((conv_b1_vis,conv_b1_sem),dim=1))

        #Decode
        deconv_b0 = F.relu(self.batchNorm0_d(self.deconv_0(conv_b6)))
        concat_0 = torch.cat((deconv_b0, conv_b5),1)

        deconv_b1 = F.relu(self.batchNorm1_d(self.deconv_1(concat_0)))
        concat_1 = torch.cat((deconv_b1, conv_b3),1)

        deconv_b2 = F.relu(self.batchNorm2_d(self.deconv_2(concat_1)))
        concat_2 = torch.cat((deconv_b2, conv_b2),1)

        deconv_b3 = F.relu(self.batchNorm3_d(self.deconv_3(concat_2)))
        concat_3 = torch.cat((deconv_b3, conv_b1),1)

        deconv_b4 = F.relu(self.batchNorm4_d(self.deconv_4(concat_3)))

        deconv_b5 = F.tanh(  self.deconv_5(deconv_b4))



        return deconv_b5,conv_b6_vis,conv_b5_vis,conv_b3_vis,conv_b2_vis,conv_b1_vis,conv_b6_sem,conv_b5_sem,conv_b3_sem,conv_b2_sem,conv_b1_sem\
        

    def decode(self,conv_b6_vis,conv_b5_vis,conv_b3_vis,conv_b2_vis,conv_b1_vis,conv_b6_sem,conv_b5_sem,conv_b3_sem,conv_b2_sem,conv_b1_sem):
       z0 =  self.combine0(torch.cat((conv_b6_vis,conv_b6_sem),dim=1))
       z =  self.combine1(torch.cat((conv_b5_vis,conv_b5_sem),dim=1))
       z2 =  self.combine2(torch.cat((conv_b3_vis,conv_b3_sem),dim=1))
       z3 =  self.combine3(torch.cat((conv_b2_vis,conv_b2_sem),dim=1))
       z4 =  self.combine4(torch.cat((conv_b1_vis,conv_b1_sem),dim=1))

       deconv_b0 = F.relu(self.batchNorm0_d(self.deconv_0(z0)))
       concat_0 = torch.cat((deconv_b0, z),1)

       deconv_b1 = F.relu(self.batchNorm1_d(self.deconv_1(concat_0)))
       concat_1 = torch.cat((deconv_b1, z2),1)
       
       deconv_b2 = F.relu(self.batchNorm2_d(self.deconv_2(concat_1)))
       concat_2 = torch.cat((deconv_b2, z3),1)

       deconv_b3 = F.relu(self.batchNorm3_d(self.deconv_3(concat_2)))
       concat_3 = torch.cat((deconv_b3, z4),1)

       deconv_b4 = F.relu(self.batchNorm4_d(self.deconv_4(concat_3)))

       deconv_b5 = F.tanh(  self.deconv_5(deconv_b4))   
       return deconv_b5     
  