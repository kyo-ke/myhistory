#デフォルトでL2正則化alpha = 1

import mglearn 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

import maplotlib.pyplot as plt


cancer= load_breast_cancer()

x_tr,x_te,y_tr,y_te = train_test_split(cancer.data,cancer.target,random_state = 42)

logreg = LogisticRegression().fit(x_tr,y_tr)

print("score {}".format(logreg.score(x_te,y_te)))

#plt で　グラフの表示、消すのを一つのコマンドラインで　スレッドでも使うのかなあ？

plt.plot(logreg.coef_.T,'o',label = "c = 1")
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation = 90)
plt.hlines(0,0,cancer.data.shape[1])
plt.ylim(-5,5)
plt.xlabel("features")
plt.ylabel("coef magnitude")
plt.legend()
plt.show()

