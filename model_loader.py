"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
from SegNet import SegNet
from VAE import VAE
##detection
from yolov3.utils import *
from yolov3.utils.utils import *
from yolov3.utils.parse_config import *
from yolov3.models import Darknet
import cv2
from conv import Conv
# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ModelLoader():
    # Fill the information for your team
    team_name = 'Team41'
    team_number = 41
    round_number = 3
    team_member = "[yt1526, df1352, nl1668, lz1883]"
    contact_email = 'nl1668@nyu.edu'

    def __init__(self, model_file1='best_roadmap.pt', model_file2 = 'best_bounding_box.pt'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.model_encoder = VAE().to(device)
        checkpoint1 = torch.load(model_file1)
        self.model_encoder.load_state_dict(checkpoint1['encode_state_dict'])
        self.model_decoder = SegNet(input_channels = 6144, output_channels = 1).to(device)
        self.model_decoder.load_state_dict(checkpoint1['decode_state_dict'])
        
        config = "./yolov3/cfg/yolov3-spp.cfg"
        hyp = {'giou': 3.54,  # giou loss gain
        'cls': 0,  # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'iou_t': 0.20,  # iou training threshold
        'lr0': 0.0001,  # initial learning rate (SGD=5E-3, Adam=5E-4)
        'lrf': 0.00005,  # final learning rate (with cos scheduler)
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.000484,  # optimizer weight decay
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
        'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
        'degrees': 1.98 * 0,  # image rotation (+/- deg)
        'translate': 0.05 * 0,  # image translation (+/- fraction)
        'scale': 0.05 * 0,  # image scale (+/- gain)
        'shear': 0.641 * 0}  # image shear (+/- deg)
        self.model = Darknet(config, verbose=False).to(device)
        checkpoint2 = torch.load(model_file2)
        self.model.load_state_dict(checkpoint2['model_state_dict'])
        self.conv = Conv().to(device)
        self.conv.load_state_dict(checkpoint2['conv_state_dict'])
        
        

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        b= []
        self.model.eval()
        self.conv.eval()
        batch_size = samples.shape[0]
        sample = samples.view(batch_size, 18, 256, 306).to(device)
        sample = self.conv(sample)
        pred = self.model(sample)
        conf_thres = 0.08
        iou_thres = 0.6
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, multi_label=False, classes=None)
        for i, p in enumerate(pred):
            pred[i][:, 0] = p[:, 0] * 80/256 - 40
            pred[i][:, 1] = p[:, 1] * 80/306 - 40
            pred[i][:, 2] = p[:, 2] * 80/256 - 40
            pred[i][:, 3] = p[:, 3] * 80/306 - 40
        box_num = pred[0].size()[0]
        bbox = []
        for i in range(box_num):
            bbox.append(torch.tensor([[max(pred[0][i][2], pred[0][i][0]),  max(pred[0][i][2], pred[0][i][0]), min(pred[0][i][2], pred[0][i][0]),  min(pred[0][i][2], pred[0][i][0])],
            [min(pred[0][i][1],pred[0][i][3]),   max(pred[0][i][1],pred[0][i][3]), max(pred[0][i][1],pred[0][i][3]), min(pred[0][i][1],pred[0][i][3])]]))
        bbox = torch.stack(bbox)
        bbox[bbox>40] = 40
        bbox[bbox<-40] = -40
        b.append(bbox)
        return tuple(b)

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        self.model_encoder.eval()
        self.model_decoder.eval()
        batch_size = samples.shape[0]
        samples = torch.unbind(samples, 1)
        encodeds = []
        for sample in samples:
            sample = sample.to(device)
            encoded = self.model_encoder.encoder(sample)
            encodeds.append(encoded)
        encode = torch.cat(encodeds, dim=1)
        output = self.model_decoder(encode)
        output_class = (output > 0.5).float()
        return output_class
