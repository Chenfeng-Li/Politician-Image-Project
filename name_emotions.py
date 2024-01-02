import numpy as np
import nltk
import pandas as pd
from PIL import Image
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
from tools import img_from_url

import Expression_Recognition.transforms as transforms
from skimage import io
from skimage.transform import resize
from Expression_Recognition.models import *



def name_cor_filter(row, names, cors):
    """
    For a row of the dataframe, check if it comes from certain corporations, 
    and if some certain strings (names) is included in the text.

    rows: row of image dataset.
    names (list): list of target strings.
    cors (list): list of the target corporations.

    Application example: find the rows of FoxNews contains "Trump"
    df.apply(name_cor_filter, axis=1, args=(["Trump"], ["foxnews", "foxbusiness"]))
    """


    
    for cor in cors:
        if row["Corporation"] == cor:
           
            # If no name given, go through every images in this corporation.
            if names == []:
                return True

            for name in names:
                if type(row['Title']) == str and name in nltk.word_tokenize(row['Title']):
                    return True
                if type(row['Caption']) == str and name in nltk.word_tokenize(row['Caption']):
                    return True
                if type(row['Text']) == str and name in nltk.word_tokenize(row['Text']):
                    return True
    return False


#### Prepare Facenet facial recognition model

try:
    model = pickle.load(open('model.sav', 'rb'))
except FileNotFoundError:
    print("Dataset model.sav not found.")
    print("Execute Embedding_faces.py first.")
    sys.exit(1) 

mtcnn = MTCNN(keep_all=True)
resnet_vggface2 = InceptionResnetV1(pretrained='vggface2').eval() 
resnet_vggface2.classify = True

name_embd = torch.load('name_embd.pt')
names = name_embd['names']

def name_from_url(img, name):
    """
    Input an image url and a certain name.
    If the image contains this name, return the crops. Otherwise, return False.

    img (PIL)
    name (list of string): exact names, included in name_embd['names']
    """
    try:
        boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
        faces = mtcnn.extract(img, boxes,save_path=None)
    except:
        return False
    
    if faces is None:
        return False
    
    for face, box in zip(faces, boxes):
        face_embd = resnet_vggface2(face.unsqueeze(0))
        face_embd = face_embd.detach()
        target = model.predict(face_embd)[0]
        # Suppose an images contains at most one face of a specific person
        for n in name:
            if n == names[target]:
                return box

    return False


def extend_limit(img):
    """ The restriction of possible crop of a PIL image. """
    a, b = img.size
    return [0, 0, a, b]

def extend_to_square(rect, limits):
    """ 
    For a crop and the restriction, extend the crop to square
    by keeping the long edge, and extend the short edge to two sides.

    rect (length-4 array): a crop
    limits (length-4 array): the restriction of the crop
    """
    left, top, right, bottom = rect
    left_lim, top_lim, right_lim, bottom_lim = limits

    width = right - left
    height = bottom - top

    if width > height:
        # Width is longer, adjust height
        target_size = width
        new_top = max(top - (target_size - height) / 2, top_lim)
        new_bottom = new_top + target_size

        # Check if bottom exceeds limit, adjust if necessary
        if new_bottom > bottom_lim:
            new_bottom = bottom_lim
            new_top = new_bottom - target_size

        return [left, new_top, right, new_bottom]

    else:
        # Height is longer, adjust width
        target_size = height
        new_left = max(left - (target_size - width) / 2, left_lim)
        new_right = new_left + target_size

        # Check if right exceeds limit, adjust if necessary
        if new_right > right_lim:
            new_right = right_lim
            new_left = new_right - target_size

        return [new_left, top, new_right, bottom]
    




### Load the facial expression recognition model
cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
net = VGG('VGG19')
checkpoint = torch.load('Expression_Recognition/model.t7', map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net'])
net.eval()



def rgb2gray(rgb):
    """ For RGB array, convert to gray scale. """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def emotion_embd(img, crop):
    """
    Imput a PIL image and a crop (probably the crop of face), pass the image crop to FER model
    return the emotion vectors.

    img (PIL): an image (with face)
    crop (length-4 array): crop coordinate of image (left, top, right, bottom).
    
    Return: 
    output_avg (length-7 array): vector (logit) stands for emotion of face 
    """
    img = img.crop(crop)
    img = np.array(img)
    gray = rgb2gray(img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)
    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    with torch.no_grad():
        outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    return outputs_avg





def name_emotion(name, mask, max_num=None):
    """
    For a certain name (included in name_embd['names']), 
    find the photos that contains this name in the masked dataframe, 
    and store the emotion vector and corresponding link of the photo.

    max_num (int or None): If integer, maximum number of emotion to get. 
                           If None, pass all photos.
    """
    emotions = []
    URLs = [] 

    if max_num is None:
        for idx, row in tqdm(df[mask].iterrows()):
            if max_num is not None and count >= max_num:
                break
            try:
                img = img_from_url(row["URL"])
                crop = name_from_url(img, name=name)
                if crop is not False:
                    crop = extend_to_square(crop, extend_limit(img))
                    emotions.append(emotion_embd(img, crop))
                    URLs.append(row["URL"])
            except:
                continue
                
    else:
        iterator = df[mask].iterrows()
        
        for i in tqdm(range(max_num)):
            while True:
                try:
                    idx, row = next(iterator)
                except:
                    # Iterator is empty
                    break

                try:
                    img = img_from_url(row["URL"])
                    crop = name_from_url(img, name=name)
                    if crop is not False:
                        crop = extend_to_square(crop, extend_limit(img))
                        emotions.append(emotion_embd(img, crop))
                        URLs.append(row["URL"])
                        break
                except:
                    continue
        
    return emotions, URLs




if __name__ == "__main__":
    try:
        df = pd.read_csv("image_dataset.csv")

    except FileNotFoundError:
        print("Dataset image_dataset.csv not found.")
        print("Execute analyze_image_dataset.py first.")
        sys.exit(1)   

    # Shuffle the dataframe
    df = df.sample(frac=1)

    try:
        politician_emotion = torch.load("politician_emotions_corporation.pt")
    except:
        politician_emotion = {}

    corporations = [["foxnews", "foxbusiness"], 
                  ["cnn"], 
                  ["washtimes", "washingtontimes"],
                  ["dailycaller"],
                  ["politico", "politicopro"],
                  ["breitbart"],
                  ["npr"],
                  ['apnews']]
    
    print("Input the exact names of politician.")
    print("Press Enter to continue.")
    # One person may have multiple exact name, 
    # e.g. "Hillary Clinton" and "Hillary Rodham Clinton"
    exact_names = []
    while True:
        n = input()
        if n == "":
            if exact_names == []:
                print("Enter at least one name of the specific person")
            else:
                break
        exact_names.append(n)
    
    print("\nInput the texts to filter.")
    print("Press Enter to continue.")
    print("If input nothing, go through every images.")
    name_texts = []
    while True:
        nt = input()
        if nt == "":
            break
        name_texts.append(nt)

    print("\nInput the maximum number of emotions to get for each corporation. 2000 by default.")
    max_num = input()
    if max_num == "":
        max_num = 2000
    else:
        max_num = int(max_num)

    
    politician_emotion[exact_names[0]] = {}

    for cor in corporations:
        print(f"Now analyzing {cor}")
        name_cor_mask = df.apply(name_cor_filter, axis=1, args=(name_texts, cor))
        emotions_list, URLs = name_emotion(exact_names, name_cor_mask, max_num=max_num)
        emotions = torch.stack(emotions_list)
        politician_emotion[exact_names[0]][cor[0]] = (emotions, URLs)
        torch.save(politician_emotion, "politician_emotions_corporation.pt")
    

