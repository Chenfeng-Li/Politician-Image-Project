import sys
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


try:
    politician_emotions = torch.load("politician_emotions_corporation.pt")
except:
    print("Emotions data politician_emotions_corporation.pt not found.")
    print("Execute name_emotions.py first.")
    sys.exit(1) 

print("Input the evaluation mode.\n 1 for full model,\n 2 if separate the media corporations by politics bias.")
mode = input()


emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
corporation_bias_ordered = ['cnn', 'npr', 'politico', 'apnews', 'washtimes', 'dailycaller', 'foxnews', 'breitbart']

if mode == 2:
    corporation_bias_ordered = ['left', 'right']


for key, val in politician_emotions.items():
    print(f"Now analyzing the emotion from {key}")
    data = []
    label = []
    num_sample = []
    l = 0

    n_left, n_right = 0,0

    for cor in corporation_bias_ordered:
        print(cor)
        v = val[cor]
        print(f"Number of Samples: {len(v[2])}")
        num_sample.append(len(v[2]))
        if len(v[2]) == 0:
            continue
        avg_logit = torch.mean(v[0],axis=0)
        avg_prob = F.softmax(avg_logit,dim=0)
        avg_prob_format = '[' + ', '.join(f"{e:.4f}" for e in avg_prob.tolist()) + ']'
        print(f"Average emotion: {avg_prob_format}")
        dominate = emotions[torch.where(avg_logit==max(avg_logit))[0][0]]
        print(f"Dominate Emotion: {dominate}")

        data.append(v[0])
        if mode == '1':
            label+= [l]*len(v[2])
        elif mode =='2':
            if cor in ['cnn', 'npr', 'politico', 'apnews']:
                label += [0]*len(v[2])
                n_left+=len(v[2])
                
            else:
                label += [1]*len(v[2])
                n_right += len(v[2])
        l+=1
        
    data = torch.cat(data,dim=0)
    X_train, X_test, y_train, y_test = train_test_split(data, label)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    if mode == '1':
        print(f"Baseline (naive choice) accuracy is {max(num_sample)/sum(num_sample)}")
    elif mode == '2':
        print(f"Baseline (naive choice) accuracy is {max(sum(num_sample[:4]),sum(num_sample[4:]))/sum(num_sample)}")
    print(f"Prediction Accuracy of KNN classification: {accuracy}\n\n")