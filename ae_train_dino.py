import time
#import utils
import presets
import datetime
from data_utils import *
from detection import *
import utils
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from sklearn import preprocessing
from torchvision import transforms as pth_transforms
#import vision_transformer as vits
import requests
import matplotlib.pyplot as plt
import json
import cv2
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from vit_utils import *
import urllib.request
from torchvision import models, transforms
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=192
        )
        self.encoder_output_layer = nn.Linear(
            in_features=192, out_features=96
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=96, out_features=192
        )
        self.decoder_output_layer = nn.Linear(
            in_features=192, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("--data-path", default="/scratch/shared/owssd_iccv/datasets/voc/", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda:1", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=1, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default="ResNet50_Weights.IMAGENET1K_V1", type=str, help="the backbone weights enum name to load")

    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    
    parser.add_argument("--annotation-file", default="train2017", type=str, help="annotation file name")
    parser.add_argument("--num-classes", default=1, type=int, help="number of classes")
    parser.add_argument("--model-class", type=int, help="model class number")
    parser.add_argument("-p", '--path', dest="imagedirpath", required =True, help='add path to image directory')
    
    return parser

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


data_transform = transforms.Compose([
   transforms.ToPILImage(),
   SquarePad(),
   transforms.Resize(98, interpolation=T.InterpolationMode.BICUBIC),
   transforms.CenterCrop(98),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        
def extract_features(model, images_map):

   features1 = []
   all_labels = []
   for image_id in images_map:
     image_map = images_map[image_id]
     file_name = image_map['file_name']
     if 'boxes' in image_map:
        boxes = image_map['boxes']
        labels = image_map['labels']
        path1 = os.path.join(args.imagedirpath, "train2017", str(file_name))
        print("path: ", path1)
        img = cv2.imread(path1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range (len(boxes)):
            box = boxes[i]
            label = labels[i]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            cropped_img = img[y:y+h, x:x+w]
            #img_tensor = torch.zeros(1, 3, 224, 224)
            img_tensor = data_transform(cropped_img)[:3].unsqueeze(0)
            features = model(img_tensor.cuda()) #384 small, #768 base, #1024 large
            features /= features.norm(dim=-1, keepdim=True)
            features = features.tolist()
            all_labels.append(label)
            features1.extend(features)
                 
 
   features2 = np.array(features1)
   #features2 = features2.astype(np.float64) 
   print("type of features :", type(features1))
   all_labels1 = np.array(all_labels)
   #print(features1, all_labels1)
   return features2, all_labels1


def create_image_map(annofile):

    with open(annofile, "r") as jsonFile:
        anno = json.load(jsonFile)
        image_list = anno['images']
        annolist = anno["annotations"]
        images_map = {}

        for image in image_list:
            image_map = {}
            image_id = image["id"]
            filename = image['file_name']
            
            if image_id not in images_map:
                images_map[image_id] = {} 
                images_map[image_id]['file_name'] = filename

        for ann in annolist:
            annid = ann["image_id"]
            image_map = images_map[annid]
            if 'boxes' not in image_map:
                image_map['boxes'] = []
                image_map['boxes'].append(ann['bbox'])
                image_map['labels'] = []
                image_map['labels'].append(ann['category_id'])
            else:
                boxes = image_map['boxes']
                boxes.append(ann['bbox'])
                image_map['boxes'] = boxes
                labels = image_map['labels']
                labels.append(ann['category_id'])
                image_map['labels'] = labels
                
        return images_map
        

def main(args):
    print(args)

    device = torch.device("cuda")
    
    vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True).to(device)
    
    vits14.eval()

    image_map = create_image_map(os.path.join(args.data_path, "annotations/instances_train2017_seen_"+str(args.model_class)+".json"))
   
    #extract features for all images
    features, labels = extract_features(vits14, image_map)
    n = len(features)
    #print("features shape :", features.shape)
    print("number of data points: ", n)
   
        
    #print("Start training")
    start_time = time.time()
    
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    ae_model = AE(input_shape=384).to(device)
   
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()
    
    for epoch in range(args.epochs):
        #for i, x in enumerate(ef_all):
        #    print(i)
        loss = 0
        
        optimizer.zero_grad()
        y = torch.from_numpy(features).to(device)
        # compute reconstructions
        outputs = ae_model(y.to(torch.float32))
            
        # compute training reconstruction loss
        train_loss = criterion(outputs, y.to(torch.float32))
            
        print("train_loss: ", train_loss)

        # compute accumulated gradients
        train_loss.backward()
            
        #perform parameter update based on current gradients
        optimizer.step()
            
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        print("loss :", loss)
        
        # compute the epoch training loss
        loss = loss / n
        print("loss divided by n : ", loss)
    
        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, args.epochs, loss))

    torch.save(ae_model.state_dict(), str("/scratch/alucic2/owssd/main/models/ae_class_new")+str(args.model_class)+str(".pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Generation time {total_time_str}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
