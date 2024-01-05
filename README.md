# Politician Image Project
Build up models to recognize the politicians in images from news websites

### Install Packages
```
pip install numpy matplotlib pandas torch Pillow requests beautifulsoup4 facenet_pytorch rmn torchvision scikit-learn pickle-mixin tqdm nltk
```
  
where 
  <ul>
  <li><code>facenet_pytorch</code> (<a href="https://github.com/timesler/facenet-pytorch">source</a>)): Key module for facial detection and recognition. </li>
  <li><code>rmn</code> (<a href="https://github.com/phamquiluan/ResidualMaskingNetwork">source</a>)): Key module for facial expression detection. </li>
  <li><code>request beautifulsoup4</code>: Extract content (links, texts, images) from website. </li>
  <li><code>scikit-learn</code>: K Nearest Neighbour classifier.</li>
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

| Corporation     | Sample Size |
|-----------------|-------------|
| Politico        | 315248      |
| DailyCaller     | 145037      |
| WashTimes       | 114970      |
| FoxNews         | 101162      |
| CNN             | 94209       |
| NPR             | 92599       |
| Breitbart       | 81336       |
| APNews          | 7715        |
| PoliticoPro     | 236         |
| FoxBusiness     | 148         |
| TheHill         | 21          |
| WP              | 13          |
| TimesOfIsrael   | 3           |
| Mediaite        | 2           |
| WashingtonTimes | 2           |
| WSJ             | 1           |
| JNS             | 1           |
| JPost           | 1           |
| AFP             | 1           |




## Facenet Model preparation

### Steps
1. Prepare another labelled dataset:<br>
   Extract names and relevant image urls of politicians from the three Wikimedia sites:
   <ul>
     <li><a href="https://commons.wikimedia.org/wiki/Category:21st-century_male_politicians_of_the_United_States">Category:21st-century male politicians of the United States</a></li>
     <li><a href="https://commons.wikimedia.org/wiki/Category:21st-century_female_politicians_of_the_United_States">Category:21st-century female politicians of the United States</a></li>
     <li><a href="https://commons.wikimedia.org/w/index.php?title=Category:21st-century_businesspeople_from_the_United_States&oldid=527515279">Category: 21st-century businesspeople from the United States</a></li>
   </ul>

   and two Wikiopedia sites:
   <ul>
     <li><a href="https://en.wikipedia.org/w/index.php?title=Category:21st-century_American_politicians&oldid=1015022478">Category:21st-century American politicians</a></li>
     <li><a href="https://en.wikipedia.org/w/index.php?title=Category:21st-century_American_businesspeople&oldid=1110690935">Category:21st-century American businesspeople</a></li>     
   </ul>

Specifically, for each personal page from Wikipedia, first check if this person has wikimedia page, and get images from wikimedia page if possible. Otherwise, take the portrait link from the Wikipedia page if there is.

```
$ python Image_URLs.py
```

This execution takes around 48 hours, producing a map "name -> list of image urls" and storing as <code>name_img_url.pt</code> (around 61 MB), which includes 312587 urls from 20707 (in particular, 10094 non-empty) persons.<br>
This file is included in the repository. 


2. For the images from <code>name_img_url.pt</code>, implementing the <code>facenet</code> functions to get embedding vectors of faces.<br>
To reduce the execution time, for each person, we take at most 10 faces embedding vectors to this dataset, where one of which are from the portrait.
```
$ python Embedding_Faces.py
```
This execution takes around 9 hours, storing the labelled vectors as <code>name_embd.pt</code> (around 1.3 GB), which includes 37934 embedding face vectors from 9308 politicians.<br>
You may also download this file from https://drive.google.com/file/d/1SLuR20JKkM4EgpwZWJkGkSI3LXy0eBkc/view?usp=share_link

3. Using K-nearest neighbour classifier (with K=1) to produce a model for the labelled vectors.
```
$ python prepare_KNN_model.py
```


### Evaluation
1. Evaluate the performance of <code>facenet</code> model with the dataset <code>name_embd.pt</code>.
   Split the train, test set at a ratio 8:2, and then train a K-nearest neighbour classifier (with K=1) model.<br>
```
$ python Evaluate_name_embd.py
```
Output:
```
>>> The prediction accuracy of name_embd dataset through 1-nearest-neighbor is 0.8059839198629234.
```


2. Test the model using a single url:

#### Example 1:
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


#### Example 2
<figure>
<img src="https://media.cnn.com/api/v1/images/stellar/prod/200505150639-02-coronavirus-task-force-briefing-0325.jpg?q=x_0,y_156,h_1688,w_3000,c_crop/w_800">
  <em>US President <b>Donald Trump</b>, flanked by (from R) Response coordinator for White House Coronavirus Task Force <b>Deborah Birx</b>, US Treasury Secretary <b>Steven Mnuchin</b>, US Vice President <b>Mike Pence</b> and Director of the National Institute of Allergy and Infectious Diseases <b>Anthony Fauci</b>, speaks during the daily briefing on the novel coronavirus, COVID-19, in the Brady Briefing Room at the White House on March 25, 2020, in Washington, DC.</em>
</figure>

Input:
```
https://media.cnn.com/api/v1/images/stellar/prod/200505150639-02-coronavirus-task-force-briefing-0325.jpg?q=x_0,y_156,h_1688,w_3000,c_crop/w_800
```
Output:
```
Recognized {'Vicki Marble', 'Donald Trump', 'Mike Pence', 'Mike Gravel', 'Steven Mnuchin'} in the image.
```

## Facial Expression Recognition (FER) model

### Source:
<a href="https://github.com/phamquiluan/ResidualMaskingNetwork">Residual Masking Network</a> from <a href="https://github.com/phamquiluan">Luan Pham</a>.

Implementing Residual Masking Network (RMN) for expression detection. Achieving 77% accuracy from <a href="https://www.kaggle.com/datasets/msambare/fer2013">Fer2013</a> dataset.





