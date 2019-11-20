import numpy as np
import pickle
from modules.pls_classifier import PLSClassifier

def labelFormat(n):
    if n == 0:
        return -1
    else:
        return n

labels = ["001", "002", "003", "004", "005", "006",
"007", "008", "009", "010"]

arq = open('HOG_files/learn.txt', 'r')

matrix_x = []
matrix_y = []

for linha in arq:
    y, x = linha.split(" | ")
    x = np.array(np.mat(x.replace("]", "").replace("[", "").strip())).ravel()
    matrix_x.append(x)
    matrix_y.append(y)
    print(y, len(x))
arq.close()

arq = open('LBP_files/learn.txt', 'r')

count = 0
for linha in arq:
    y, x = linha.split(" | ")
    x = np.array(np.mat(x.replace("]", "").replace("[", "").strip())).ravel()
    matrix_x[count] = np.concatenate((matrix_x[count], x), axis=0)
    print(y, len(matrix_x[count]))
    count += 1
arq.close()

matrix_y = np.array(matrix_y)
models = []
for label in labels:
    classifier = PLSClassifier()
    y = np.char.count(matrix_y, label)
    y = list(map(labelFormat, y))
    model = classifier.fit(np.array(matrix_x), np.array(y))
    models.append(model)

path = "models_pls/hog_lbp/"
name_arq = input("Template Name: ")
pickle.dump(models, open(path+name_arq+".sav", 'wb'))
