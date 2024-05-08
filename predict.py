import json
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('special_file', action='store', default='unique_special.pth')
    parser.add_argument('--k_top', dest='k_top', default='3')
    parser.add_argument('--file_path', dest='file_path', default='flowers/test/5/image_05159.jpg')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def image_processing(image_file):
    img_pil = Image.open(image_file)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = adjustments(img_pil)
    return image

def load_special_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')

    # Define the model
    model = models.resnet50(pretrained=True)

    # Load the state_dict into the model
    model.load_state_dict(checkpoint['extraordinary_model'])

    # Load the classifier
    model.fc = checkpoint['extraordinary_classifier']

    # Load the class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def fetch_special_category_names(filename):
    with open(filename) as f:
        extraordinary_category_names = json.load(f)
    return extraordinary_category_names

def special_prediction(image_path, special_model, top_k=3, gpu='gpu'):
    if gpu == 'gpu' and torch.cuda.is_available():
        special_model = special_model.cuda()
    else:
        special_model = special_model.cpu()
        
    img_torch = image_processing(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if gpu == 'gpu' and torch.cuda.is_available():
        with torch.no_grad():
            output = special_model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = special_model.forward(img_torch)
        
    probability = F.softmax(output.data, dim=1)
    probs = np.array(probability.topk(top_k)[0][0])
    
    index_to_class = {val: key for key, val in special_model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(top_k)[1][0])]
    
    return probs, top_classes

def main_execution(): 
    args = parse_input()
    gpu = args.gpu
    special_model = load_special_checkpoint(args.special_file)
    cat_to_name = fetch_special_category_names(args.category_names)
    
    img_path = args.file_path
    probs, classes = special_prediction(img_path, special_model, int(args.k_top), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + img_path)
    
    print(labels)
    print(probability)
    
    i = 0
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1

if __name__ == "__main__":
    main_execution()