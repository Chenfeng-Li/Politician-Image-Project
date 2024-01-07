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


def embedding_face(name, max_num=torch.inf):
    """
    For a name from name_img_url, obtain the embedding face from FaceNet, and 
    """
    # Initialize name_embd[name]
    try:
        del name_embd[name]
    except:
        pass
    
    urls = name_img_url[name]
    
    if len(urls) == 0: # No urls
        return
    
    # Embedding faces (Matrix) of the person
    embd = []
    
    controlled_embd = None

    # Almost each person has a portrait from the first few urls, which has only one face.
    # Find the portrait as "controlled face".
    for label, url in enumerate(urls):
        try:
            img = img_from_url(url)       
            if img.mode == "P":
                img = img.convert("RGBA")
            img = img.convert("RGB")
            boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
            faces = mtcnn.extract(img, boxes,save_path=None)
        except:
            continue
            
        if faces is None or len(faces) > 1: # No face or more than one faces:
            continue
        controlled_embd = resnet(faces[0].unsqueeze(0))
        controlled_embd = controlled_embd.detach()
        
        embd.append(controlled_embd)
        url_crops['urls'].append(url)
        url_crops['crops'].append(boxes[0])
        
        break
    
    if controlled_embd is None: # No controlled face
        return
    
    rest_urls = urls[(label+1):]
    shuffle(rest_urls)

    # Find faces close to the controlled face
    for url in rest_urls:
        
        if len(embd) >= max_num:
            break

        try:
            img = img_from_url(url)
            if img.mode == "P":
                img = img.convert("RGBA")
            img = img.convert("RGB")
            boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
            faces = mtcnn.extract(img, boxes,save_path=None)
        except:
            continue

        if faces is None: #No face
            continue

        # Expect at most one face of a specific person in an image
        closest_face_label = None
        closest_face_embd = None
        closest_dist = torch.inf
        
        for i, face in enumerate(faces):
            face_embd = resnet(face.unsqueeze(0))
            face_embd = face_embd.detach()
            
            dist = torch.norm(face_embd - controlled_embd)
            if dist < closest_dist:
                closest_dist = dist
                closest_face_label = i
                closest_face_embd = face_embd
                
        if closest_face_label is None:
            continue
            
        if closest_dist < 200: # This closest face is similar to controlled face. Take 200 as threshold
            embd.append(closest_face_embd)
            url_crops['urls'].append(url)
            url_crops['crops'].append(boxes[closest_face_label])
            
                
    embd = torch.tensor(embd)
    name_embd[name] = embd
                
                

if __name__ == "__main__":

    try:
        name_img_url = torch.load("name_img_url.pt")
    except FileNotFoundError:
        print("name_img_url.pt not found. Run Image_URLs_Wikimedia.py first")
        sys.exit(1)
        
    # Resume or initialize name_embd dictionary

    name_embd = {}
    url_crops = {'urls':[], 'crops': []}

    # Detection
    mtcnn = MTCNN(keep_all=True)

    # Embedding
    resnet = InceptionResnetV1(pretrained='vggface2').eval() 
    resnet.classify = True

    # Takes around 9 hours
    for name in tqdm(name_img_url.keys(),position=0, leave=True):
        embedding_face(name, max_num=10)

    # Reconstruct the name_embd dataset
    X = torch.cat(tuple(name_embd.values()))
    names = list(name_embd.keys())
    y = []
    for i, val in enumerate(name_embd.values()):
        y += [i]*len(val)
    y = torch.tensor(y)

    name_embd = {'names': names,
                'embd_faces': X,
                'targets': y,
                'urls': url_crops['urls'],
                'crops': url_crops['crops']
                }

    # Save the embedding faces
    # Size: 1.32 GB
    torch.save(name_embd,"name_embd.pt")

