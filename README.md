# Politician Image Project
Build up models to recognize the politicians in images from news websites

## Dataset
https://drive.google.com/file/d/1QWZ5JAdihzDzdTQX2Xjk4rFNm_ixurz7/view?usp=drive_link

## Model preparation

### Install Packages
```
pip install numpy matplotlib pandas torch Pillow requests beautifulsoup4 facenet_pytorch torchvision scikit-learn pickle-mixin tqdm
```
  
where 
  <ul>
  <li><code>facenet_pytorch</code> (<a href="https://github.com/timesler/facenet-pytorch">source</a>)): Key module for facial detection and recognition. </li>
  <li><code>request beautifulsoup4</code>: Extract content (links, texts, images) from website. </li>
  <li><code>Pillow</code>: Python image library.</li>
  <li><code>scikit-learn</code>: Import K Nearest Neighbour classifier.</li>
  <li><code>pickle-mixin</code>: Save sklearn model.</li>
  </ul>

## Steps
1. Prepare another labelled dataset:
   Extract names and relevant image urls of politicians from the two Wikimedia sites:
   <ul>
     <li><a href="[Category:21st-century male politicians of the United States](https://commons.wikimedia.org/wiki/Category:21st-century_male_politicians_of_the_United_States)https://commons.wikimedia.org/wiki/Category:21st-century_male_politicians_of_the_United_States">Category:21st-century male politicians of the United States</a></li>
     <li><a href="https://commons.wikimedia.org/wiki/Category:21st-century_female_politicians_of_the_United_States">Category:21st-century female politicians of the United States</a></li>
   </ul>
```
$ python Image_URLs_Wikimedia.py
```
This execution takes around 24 hours, producing a map "name -> list of image urls" and storing as <code>name_img_media.pt</code> (around 32 MB).
You may also download this file from https://drive.google.com/file/d/1jTEG2ckG4MUfbpYp4NZLCGyoK5EnPvHS/view?usp=share_link

2. For the images from <code>name_img_media.pt</code>, implementing the <code>facenet</code> functions to get embedding vectors of faces.
```
$ python Embedding_Faces.py
```
This execution takes around 36 hours, storing the labelled vectors as <code>name_embd.pt</code>.
You may also download this file from https://drive.google.com/file/d/1MQiAbFaqVBLmQ6Z7yMtLj950II4kbqmz/view?usp=share_link

3. Using K-nearest neighbour classifier to produce a model for the labelled vectors.
```
$ python prepare_model.py
```


