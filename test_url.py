import sys
import pickle
import torch
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1

from tools import img_from_url

model = pickle.load(open('model.sav', 'rb'))

mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval() 
resnet.classify = True

name_embd = torch.load('name_embd.pt')
names = name_embd['names']
targets = name_embd['targets']

def name_from_url(url):
    """
    Input an image url, return the list of name of politicians.
    """
    try:
        img = img_from_url(url)
        img = img.convert('RGB')
        faces = mtcnn(img)
    except:
        return "Invalid URL"
    
    if faces is None:
        return set()
    
    ns = []
    for face in faces:
        face_embd = resnet(face.unsqueeze(0))
        face_embd = face_embd.detach()

        dist, idx = model.kneighbors(face_embd, n_neighbors=1)
        if dist < 200:
            target = targets[idx][0]
            ns.append(names[target])
    return set(ns)

if __name__ == "__main__":
    print('Enter to exit the program.')
    while True:
        print("Input an image url:")
        url = input()
        if url == "":
            break

        ns = name_from_url(url)

        if not ns:
            print("No politicians in the image.")
        elif ns == "Invalid URL":
            print(ns)
        else:
            print(f"Recognized {ns} in the image.")
