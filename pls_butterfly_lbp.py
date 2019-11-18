import numpy as np
import pickle
from modules.pls_classifier import PLSClassifier

def labelFormat(n):
    if n == 0:
        return -1
    else:
        return n
arq = open('LBP_files/learn.txt', 'r')

matrix_x = []
matrix_y = []

for linha in arq:
    y, x = linha.split(" | ")
    x = np.array(np.mat(x.strip("[]").strip())).ravel()
    print(len(x))
    matrix_x.append(x)
    matrix_y.append(y)
arq.close()

matrix_y = np.array(matrix_y)
labels = ["001", "002", "003", "004", "005", "006",
"007", "008", "009", "010"]

models = []
for label in labels:
    classifier = PLSClassifier()
    y = np.char.count(matrix_y, label)
    y = list(map(labelFormat, y))
    model = classifier.fit(np.array(matrix_x), np.array(y))
    models.append(model)

path = "models_pls/lbp/"
name_arq = input("Nome do modelo: ")
pickle.dump(models, open(path+name_arq+".sav", 'wb'))
# arq = open('LBP_files/test.txt', 'r')
#
# matrix_x = []
# matrix_y = []
#
# for linha in arq:
#     y, x = linha.split(" | ")
#     x = np.array(np.mat(x.strip("[]").strip())).ravel()
#     matrix_x.append(x)
#     matrix_y.append(y)
# arq.close()
#
# acc = 0
# for i in range(len(matrix_y)):
#     resp = []
#     for model in models:
#         predictValue = model.predict_confidence(matrix_x[i])
#         resp.append(predictValue)
#     idx = resp.index(max(resp))
#     print(matrix_y[i], labels[idx], resp[idx])
#     if matrix_y[i] == labels[idx]:
#         acc += 1
# print("Acertos: "+str(acc)+" de "+str(len(matrix_y)))
