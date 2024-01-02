import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

name_embd = torch.load("name_embd.pt")
X = name_embd['embd_faces']
y = name_embd['targets']

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

# Save the model
pickle.dump(model, open('model.sav', 'wb'))