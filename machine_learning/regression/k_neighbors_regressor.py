#回帰分析
#k-neighbors regressor

#データ型の取得コマンド
import mglearn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

x,y = mglearn.datasets.make_wave(n_samples = 40)

x_tr, x_te,y_tr,y_te = train_test_split(x,y,random_state = 0)
reg = KNeighborsRegressor(n_neighbors = 3)
reg.fit(x_tr,y_tr)

print("test set prediction {}".format(reg.predict(x_te)))