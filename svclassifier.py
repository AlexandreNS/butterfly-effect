import numpy as np
from sklearn.svm import SVC

arq = open('HOG_files/learn.txt', 'r')

matrix_x = []
matrix_y = []

for linha in arq:
    y, x = linha.split(" | ")
    x = np.array(np.mat(x.replace("]", "").replace("[", "").strip())).ravel()
    matrix_x.append(x)
    matrix_y.append(y)
    print(len(x))
arq.close()

arq = open('LBP_files/learn.txt', 'r')

count = 0
for linha in arq:
    y, x = linha.split(" | ")
    x = np.array(np.mat(x.replace("]", "").replace("[", "").strip())).ravel()
    matrix_x[count] = np.concatenate((matrix_x[count], x), axis=0)
    print(len(matrix_x[count]))
    count += 1
arq.close()

matrix_y = np.array(matrix_y)
labels = ["001", "002", "003", "004", "005", "006",
"007", "008", "009", "010"]


svclassifier = SVC(kernel='linear')
svclassifier.fit(matrix_x, matrix_y)

arq = open('HOG_files/test.txt', 'r')

matrix_x = []
matrix_y = []

for linha in arq:
    y, x = linha.split(" | ")
    x = np.array(np.mat(x.replace("]", "").replace("[", "").strip())).ravel()
    matrix_x.append(x)
    matrix_y.append(y)
arq.close()

arq = open('LBP_files/test.txt', 'r')

count = 0
for linha in arq:
    y, x = linha.split(" | ")
    x = np.array(np.mat(x.replace("]", "").replace("[", "").strip())).ravel()
    matrix_x[count] = np.concatenate((matrix_x[count], x), axis=0)
    print(len(matrix_x[count]))
    count += 1
arq.close()

acc = 0
for i in range(len(matrix_y)):
    resp = []
    predictValue = svclassifier.predict(matrix_x[i].reshape(1, -1))
    print(matrix_y[i], predictValue[0])
    if matrix_y[i] == predictValue[0]:
        acc += 1
print("Acertos: "+str(acc)+" de "+str(len(matrix_y)))
