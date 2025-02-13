from cgi import test
import time
import datetime
import pickle
import json
import argparse
import errno
import os
import math
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torchvision.transforms import transforms as T
from torchvision.datasets import CocoDetection
import pandas as pd
import numpy as np
from numpy import quantile
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score

class OWSSDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img = self._transforms(img)
        return img, target, image_id

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=192
        )
        self.encoder_output_layer = nn.Linear(
            in_features=192, out_features=64
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=64, out_features=192
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
    parser.add_argument("--data-path", default="/bbgp/garvita4/open-ubteacher/unbiased-teacher/datasets/coco", type=str, help="dataset path")
    parser.add_argument("--json-dir", default="/bbgp/garvita4/open-ubteacher/unbiased-teacher/datasets/coco/annotations", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda:0", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--epochs", default=20, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--annotation-file", default="val2017", type=str, help="annotation file name")
    parser.add_argument("--num-classes", default=1, type=int, help="number of classes")
    parser.add_argument("--model-class", type=int, help="model class number")
    return parser

def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_dataset(image_set, file_name, transform, root, prefix="instances"):
    img_folder = os.path.join(root, image_set)
    anno_file_template = "instances_{}.json"
    ann_file = os.path.join("annotations", anno_file_template.format(file_name))
    print ("Annotation file: ", ann_file)
    ann_file = os.path.join(root, ann_file)
    dataset = OWSSDetection(img_folder, ann_file, transforms=transform)
    return dataset

def extract_features(model_dino, images, targets, device, seen):
    features = []
    binary = []

    for img, target in zip(images, targets):
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        categories = [obj["category_id"] for obj in anno]

        for cat in categories:
            if cat in seen: 
                binary.append(0)
                print("binary: 0")
            else:
                binary.append(1)
                print("binary: 1")

        crop_transform = T.Compose([
                            T.ToPILImage(),
                            SquarePad(),
                            T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
            
        for bbox in boxes:
            with torch.no_grad():
                cropped_img = F.crop(img, math.floor(bbox[0]), math.floor(bbox[1]), math.floor(bbox[3]), math.floor(bbox[2]))
                cropped_img = crop_transform(cropped_img)
                cropped_img = cropped_img.reshape(1, 3, 224, 224)
                feature = model_dino(cropped_img.to(device))
                features.append(feature)
        
    features = torch.stack(features, dim=0)

    return features, binary

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def get_affinity(testfile, all_labels):
    af = AffinityPropagation(preference=-600, random_state=0).fit(testfile)
    cluster_centers_indices = af.cluster_centers_indices_
    af_labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(all_labels, af_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(all_labels, af_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(all_labels, af_labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(all_labels, af_labels))
    print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(all_labels, af_labels)
    )
    print(
    "Silhouette Coefficient: %0.3f"
    % metrics.silhouette_score(testfile, af_labels, metric="sqeuclidean")
    )
    return af_labels   

def main(args):
    if args.output_dir:
        mkdir(args.output_dir)

    #init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    
    # Data loading code
    print("Loading data")

    transform = T.Compose(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )

    dataset_test = get_dataset("val2017", args.annotation_file, transform, args.data_path)

    print("Creating data loaders")
   
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, sampler=test_sampler, num_workers=args.workers, collate_fn=collate_fn
    )
    print("Length of dataset: ", len(data_loader_test))
    print("Creating model")
    
    #model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #model_dino.head = nn.Linear(384, 1).to(device)
    model_dino.to(device)
    model_dino.eval()

    print("Start testing")
    start_time = time.time()
    
    scaler = MinMaxScaler()

    with open('thresholds.pkl', 'rb') as f:
        thresholds = pickle.load(f)

    seen = [5,7, 15, 17,18, 21, 23, 25, 31, 35, 48, 57, 58, 59, 63, 76, 77, 82, 87, 90]
    #seen = [90,76,5]
    binary = []
    predictions = []

    for images, targets, idx in data_loader_test:
        print("here")
        images = list(img.to(device) for img in images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        features, binary_ind = extract_features(model_dino, images, targets, device, seen)
        binary.extend(binary_ind)
        
        al = len(features)
        #print(centers)
        #print("len(features): ",len(features[0]))
        columns=[]
        for i in range(al):
            #i = ''.join(['c', str(i)])
            columns.append(i)
        
        l = []
        criterion = nn.MSELoss()
        #centers = torch.from_numpy(centers)
        
        inl = []
        for feature in features:
            for j in seen:
                ae_model = AE(input_shape=384).to(device)
                #ae_model.load_state_dict(torch.load("ae_"+str(j)+str(".pth"), map_location='cuda:1'))
                ae_model.load_state_dict(torch.load("ae_"+str(j)+str(".pth")))
                ae_model = ae_model.to(device)
                #print(ae_model)
                #feature = torch.from_numpy(features[i])
                feature = feature.to(device)
                ae_model.eval()
                outputs = ae_model(feature)
                score = criterion(outputs, feature).data.item()
                print("score: ", score)
                print("threshold: ", thresholds[str(j)])

                if score < thresholds[str(j)]:
                    inl.append(1)
                else:
                    inl.append(0)
        
        print("inl: ", inl)    
        if 1 in inl:
            predictions.append(0)
            print("prediction: 0")
        else: 
            predictions.append(1)
            print("prediction: 1")
            


    print("len binary: ", len(binary))
    print("len predictions: ", len(predictions))

    tpr = recall_score(binary, predictions)
    tnr = recall_score(binary, predictions, pos_label = 0) 
    fpr = 1 - tnr
    fnr = 1 - tpr
    print("fpr :", fpr)
    print("fnr :", fnr) 

    auroc = metrics.roc_auc_score(binary, predictions)
    print("auroc binary :", auroc)
    print("area under curve (auc) binary: ", auroc)
    print("fpr binary:", fpr)
    print("tpr binary :", tpr)

    
    threshold = 0.08
    score_voc_min['scores'] = score_voc_min['min_score_voc'].gt(threshold).astype(int)
    scores = score_voc_min['scores'].tolist()
    scoresb = [1 if i>0.08 else 0 for i in scores]

    
    li = []
    #extracting clusters with a maximum score_coco of 0.5, no class gave the cluster a higher score_coco
    
    print("Number of retrieved data points :", len(li))
    cl = score_voc_min.query('min_score_voc>0.08')['anno_labels']
    print("cl :", cl)
    for i in cl:
        #i = str(i).replace("c", "")
        li.append(int(i))
    
    print("li :", li)
      
    anno_ids = np.array(anno_ids)
    anno_ids1 = anno_ids[cl]
    print("anno_ids1 len :" , len(anno_ids1))

    img_ids = np.array(img_ids)
    img_ids1 = img_ids[cl]
    print("img_id1s len :" , len(img_ids1))
    
    anno_labels = np.array(anno_labels)
    anno_labels1 = anno_labels[cl]
    print("anno_labels1 len : ", len(anno_labels1))
    
              
    tpr = recall_score(binary, scoresb)
    tnr = recall_score(binary, scoresb, pos_label = 0) 
    fpr = 1 - tnr
    fnr = 1 - tpr
    print("fpr :", fpr)
    print("fnr :", fnr) 
               
    
    auroc = metrics.roc_auc_score(binary, scoresb)
    print("auroc binary :", auroc)
    print("area under curve (auc) binary: ", metrics.roc_auc_score(binary, scoresb))
    print("fpr binary:", fpr)
    print("tpr binary :", tpr)
    
    np.save('outlier_anno_ids_similar_oracle_t1_seed2', anno_ids1)
    np.save('outlier_labels_similar_oracle_t1_seed2', anno_labels1)
    np.save('outlier_imgids_similar_oracle_t1_seed2', img_ids1)
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Generation time {total_time_str}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
