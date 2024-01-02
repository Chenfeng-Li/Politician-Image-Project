"""
Evaluate the name_embd dataset.
Split name_embd into train set and test set, 
then implement a K-Nearest Neighbor classifier with K = 1 on the dataset. 
Print the accuracy.
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

name_embd = torch.load("name_embd.pt")
X = name_embd['embd_faces']
y = name_embd['targets']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

predict = model.predict(X_test)
is_acc = predict == np.array(y_test)
acc = sum(is_acc)/len(is_acc)
print(f"The prediction accuracy of name_embd dataset through 1-nearest-neighbor is {acc}.")
