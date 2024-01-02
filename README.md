# Politician Image Project
Build up models to recognize the politicians in images from news websites

### Install Packages
```
pip install numpy matplotlib pandas torch Pillow requests beautifulsoup4 facenet_pytorch torchvision scikit-learn pickle-mixin tqdm nltk
```
  
where 
  <ul>
  <li><code>facenet_pytorch</code> (<a href="https://github.com/timesler/facenet-pytorch">source</a>)): Key module for facial detection and recognition. </li>
  <li><code>request beautifulsoup4</code>: Extract content (links, texts, images) from website. </li>
  <li><code>Pillow</code>: Python image library.</li>
  <li><code>scikit-learn</code>: Import K Nearest Neighbour classifier.</li>
  <li><code>pickle-mixin</code>: Save sklearn model.</li>
  <li><code>nltk</code>: Splits English sentence to words.</li>
  </ul>



## Dataset
https://drive.google.com/file/d/1QWZ5JAdihzDzdTQX2Xjk4rFNm_ixurz7/view?usp=drive_link

## Analyze and processing the dataset
```
$ python analyze_image_dataset.py
```
Rename the columns. Delete lines with no URLs, invalid URLs, or duplicates.<br>
Determine the source corporation of each row by analyzing the URLs. Remove the rows from invalid or non-media corporation.

After processing, the sample size of dataset is **952705**, where the number of samples from each corporation is:

| Politico       | 315248 |
| DailyCaller    | 145037 |
| WashTimes      | 114970 |
| FoxNews        | 101162 |
| CNN            | 94209  |
| NPR            | 92599  |
| Breitbart      | 81336  |
| APNews         | 7715   |
| PoliticoPro    | 236    |
| FoxBusiness    | 148    |
| TheHill        | 21     |
| WP             | 13     |
| TimesOfIsrael  | 3      |
| Mediaite       | 2      |
| WashingtonTimes| 2      |
| WSJ            | 1      |
| JNS            | 1      |
| JPost          | 1      |
| AFP            | 1      |




## Model preparation



### Steps
1. Prepare another labelled dataset:<br>
   Extract names and relevant image urls of politicians from the two Wikimedia sites:
   <ul>
     <li><a href="https://commons.wikimedia.org/wiki/Category:21st-century_male_politicians_of_the_United_States">Category:21st-century male politicians of the United States</a></li>
     <li><a href="https://commons.wikimedia.org/wiki/Category:21st-century_female_politicians_of_the_United_States">Category:21st-century female politicians of the United States</a></li>
   </ul>
```
$ python Image_URLs_Wikimedia.py
```
This execution takes around 24 hours, producing a map "name -> list of image urls" and storing as <code>name_img_media.pt</code> (around 32 MB), which includes 165546 urls from 2265 politicians.<br>
You may also download this file from https://drive.google.com/file/d/1jTEG2ckG4MUfbpYp4NZLCGyoK5EnPvHS/view?usp=share_link

2. For the images from <code>name_img_media.pt</code>, implementing the <code>facenet</code> functions to get embedding vectors of faces.
```
$ python Embedding_Faces.py
```
This execution takes around 36 hours, storing the labelled vectors as <code>name_embd.pt</code> (around 3.2 GB), which includes 91186 embedding face vectors from 2182 politicians.<br>
You may also download this file from https://drive.google.com/file/d/1MQiAbFaqVBLmQ6Z7yMtLj950II4kbqmz/view?usp=share_link

3. Using K-nearest neighbour classifier (with K=1) to produce a model for the labelled vectors.
```
$ python prepare_model.py
```


## Evaluation
1. Evaluate the performance of <code>facenet</code> model with the dataset <code>name_embd.pt</code>.
   Split the train, test set at a ratio 8:2, and then train a K-nearest neighbour classifier (with K=1) model.<br>
```
$ python Evaluate_name_embd.py
```
Output:
```
>>> The prediction accuracy of name_embd dataset through 1-nearest-neighbor is 0.9524618927513981.
```

2. Test the model using a single url:

We are interesting in the two persons from the image as follows <br>
<figure>
<img src="https://www.politico.com/dims4/default/210ea62/2147483647/strip/true/crop/700x400+0+0/resize/630x360!/quality/90/?url=https%3A%2F%2Fstatic.politico.com%2Fcapny%2Fsites%2Fdefault%2Ffiles%2Fa-Kirsten%20Gillibrand-Chuck%20Schumer_0.png">
  <em>https://www.politico.com/dims4/default/210ea62/2147483647/strip/true/crop/700x400+0+0/resize/630x360!/quality/90/?url=https%3A%2F%2Fstatic.politico.com%2Fcapny%2Fsites%2Fdefault%2Ffiles%2Fa-Kirsten%20Gillibrand-Chuck%20Schumer_0.png</em>
</figure>

```
$ python test_url.py
```

Input:
```
https://www.politico.com/dims4/default/210ea62/2147483647/strip/true/crop/700x400+0+0/resize/630x360!/quality/90/?url=https%3A%2F%2Fstatic.politico.com%2Fcapny%2Fsites%2Fdefault%2Ffiles%2Fa-Kirsten%20Gillibrand-Chuck%20Schumer_0.png
```

Output:
```
>>> Recognized {'Kirsten Gillibrand', 'Charles Schumer'} in the image.
```
which are exactly the names of the two politicians.




