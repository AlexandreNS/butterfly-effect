import numpy as np
import pickle
import matplotlib.pyplot as plt
from functools import reduce

path = "models_pls/hog/"
name_arq = input("Nome do modelo: ")
models = pickle.load(open(path+name_arq+".sav", 'rb'))

labels = ["001", "002", "003", "004", "005", "006",
"007", "008", "009", "010"]

arq = open('HOG_files/test.txt', 'r')

matrix_x = []
matrix_y = []

for linha in arq:
    y, x = linha.split(" | ")
    x = np.array(np.mat(x.replace("]", "").replace("[", "").strip())).ravel()
    matrix_x.append(x)
    matrix_y.append(y)
arq.close()

acc = 0
clust_data = np.zeros((10,10))
for i in range(len(matrix_y)):
    resp = []
    for model in models:
        predictValue = model.predict_confidence(matrix_x[i])
        resp.append(predictValue)
    idx = resp.index(max(resp))
    print(matrix_y[i], labels[idx], resp[idx])
    clust_data[labels.index(matrix_y[i]), idx] += 1
    if matrix_y[i] == labels[idx]:
        acc += 1
print("Acertos: "+str(acc)+" de "+str(len(matrix_y)))

cellColours = []
rowcollColours = []
for i in range(10):
    sum_imgs = reduce((lambda x, y: x + y), clust_data[i])
    clust_data[i] = clust_data[i]/sum_imgs*100
    clust_data[i] = np.array(list(map(lambda a : round(a,1), clust_data[i])))
    rowcollColours.append( (209/255, 196/255, 233/255) )
    line = []
    for j in range(10):
        if i == j:
            line.append( (129/255, 199/255, 132/255) )
        else:
            line.append( (1, 1, 1) )
    cellColours.append(line)

arq = open(path+name_arq+".txt", "r")

fig, axs = plt.subplots(1)
fig.suptitle("".join(arq.readlines()).strip())

arq.close()

collabel = rowlabel = ("001", "002", "003", "004", "005", "006", "007", "008", "009", "010")
axs.axis('tight')
axs.axis('off')
the_table = axs.table(rowColours=rowcollColours,colColours=rowcollColours,cellColours=cellColours,
                    cellText=clust_data,colLabels=collabel,rowLabels=rowlabel,loc='center')
plt.show()
