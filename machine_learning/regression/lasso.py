#L１正則化
#いくつかの特徴量が完全に0になる。
#alpha は正則化の強さ

from sklearn.linear_model import Lasso
import mglearn 
import numpy as np
from sklearn.model_selection import train_test_split

x,y = mglearn.datasets.load_extended_boston()

x_tr,x_te,y_tr,y_te = train_test_split(x,y,random_state = 0)

lasso = Lasso(alpha  = 0.01).fit(x_tr,y_tr)

print("lasso score {}".format(lasso.score(x_te,y_te)))

print("number  of features used {}".format(np.sum(lasso.coef_ != 0)))

