
import numpy as np
from torch import nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url



def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        # print (size)
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]

        return x

def load_adv_imagenet(model_list,device=None,print_m=True):

    nets = []
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for model_name in model_list:
        if print_m:
            print(model_name)
        if model_name == "VGG16":
            pretrained_model = models.vgg16_bn(pretrained=True)
        elif model_name == 'Resnet18':
            pretrained_model = models.resnet18(pretrained=True)
        elif model_name == 'Squeezenet':
            pretrained_model = models.squeezenet1_1(pretrained=True)
        elif model_name == 'Googlenet':
            pretrained_model = models.googlenet(pretrained=True)
        elif model_name =='resnet50':
            pretrained_model = models.resnet50(pretrained=True)
        elif model_name == "ConvNextBase":
            pretrained_model = models.convnext_base(weights="IMAGENET1K_V1")
        elif model_name == "EfficientB3":
            from torchvision.models._api import WeightsEnum
            WeightsEnum.get_state_dict = get_state_dict 
            pretrained_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        elif model_name == "SwinV2T":
            pretrained_model = models.swin_v2_t(weights="IMAGENET1K_V1")
        elif model_name == "Resnet101":
            pretrained_model = models.resnet101(weights="IMAGENET1K_V2")

        
        else:
            print(f"model not found:{model_name}")
            return
       
        net = nn.Sequential(
            Normalize(mean, std),
            pretrained_model
        )
        nets.append(net)

    for i in range(len(nets)):
        if device is None:
            nets[i] = nets[i].cuda()
        else:
            nets[i] = nets[i].to(device)
        nets[i].eval()
        
    return nets

