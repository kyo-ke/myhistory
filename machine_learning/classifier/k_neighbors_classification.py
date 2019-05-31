import numpy as np
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.neighbors import KNeighborsClassifier
x,y = mglearn.datasets.make_forge()

x_tr, x_te, y_tr, y_te = train_test_split(x,y,random_state = 0)

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(x_tr,y_tr)

#テストデータの予測
print("test set prediction : {}".format(clf.predict(x_te)))

print("test set score {}".format(clf.score(x_te,y_te)))


