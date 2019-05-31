#ridge 回帰　線形回帰モデルの一種
#重みの値をなるべく小さく(コスト関数に重みも入れる)
#正則化
#alpha は正則化の強さ
#L2正則化(二乗ノルム)


from sklearn.linear_model import Ridge
import mglearn 
import numpy as np
from sklearn.model_selection import train_test_split

x,y = mglearn.datasets.make_wave(n_samples = 60)

x_tr,x_te,y_tr,y_te = train_test_split(x,y,random_state = 0)

a = 1.0

ridge = Ridge(alpha = a).fit(x_tr,y_tr)

print("coef_ : {}".format(ridge.coef_))

print("intercept : {}".format(ridge.intercept))

print("ridge score {}".format(ridge.score(x_te,y_te))

ridge.predict(x_te)
