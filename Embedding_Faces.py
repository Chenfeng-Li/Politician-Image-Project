"""
Fetch images from url.
Then pass each image to Facenet detection and recognition model, to get the embedding vectors of faces.
"""

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import ImageFile
from tqdm import tqdm

from tools import img_from_url

ImageFile.LOAD_TRUNCATED_IMAGES = True


def embedding_face(name):
    # Initialize name_embd[name]
    try:
        del name_embd[name]
    except:
        pass
    
    urls = name_img_media[name]
    
    if len(urls) == 0: # No urls
        return
    
    controlled_embd = None

    # Almost each person has a portrait from the first few urls, which has only one face.
    # Find the portrait as "controlled face".
    for url in urls:
        img = img_from_url(url)       
        img = img.convert('RGB')
        faces = mtcnn(img)
        if faces is None or len(faces) > 1: # No face or more than one faces:
            continue
        controlled_embd = resnet(faces[0].unsqueeze(0))
        controlled_embd = controlled_embd.detach()
        break
    
    if controlled_embd is None: # No controlled face
        return
    
    # Embedding faces (Matrix) of the person
    embd = []
    
    # Find faces close to the controlled face
    for url in urls:
        try:
            img = img_from_url(url)
            img = img.convert('RGB')
            faces = mtcnn(img)
        except:
            continue
        if faces is None: #No face
            continue
        
        # Expect at most one face of a specific person in an image
        similar_face_embd = []
        dists = []
        
        for face in faces:
            face_embd = resnet(face.unsqueeze(0))
            face_embd = face_embd.detach()
            
            dist = torch.norm(face_embd - controlled_embd)
            if dist < 200: # This face is similar to controlled face
                similar_face_embd.append(face_embd)
                dists.append(dist)
        
        if len(similar_face_embd) == 0:
            continue
        most_similar = dists.index(min(dists))
        embd.append(similar_face_embd[most_similar])
                
    embd = torch.cat(embd)
    name_embd[name] = embd
                

if __name__ == "__main__":

    # Resume the name_img_media map
    name_img_media = torch.load("name_img_media")

    # Detection
    mtcnn = MTCNN(keep_all=True)

    # Embedding
    resnet = InceptionResnetV1(pretrained='vggface2').eval() 
    resnet.classify = True

    name_embd = {}
    # Takes around 36 hours
    for name in tqdm(name_img_media.keys(),position=0, leave=True):
        embedding_face(name)

    # Reconstruct the name_embd dataset
    X = torch.cat(tuple(name_embd.values()))
    names = list(name_embd.keys())
    y = []
    for i, val in enumerate(name_embd.values()):
        y += [i]*len(val)
    y = torch.tensor(y)

    name_embd = {'Embd_faces': X,
                 'names': names,
                 'target': y}

    # Save the embedding faces
    # Size: 3.15 GB
    torch.save(name_embd,"name_embd.pt")

