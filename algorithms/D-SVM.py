import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from sklearn import preprocessing


data_test = pd.read_csv("real example-testdata.csv")
print(data_test.shape)
data_test = data_test.iloc[0:data_test.shape[0], 1:]
y_test = np.array(data_test)
print(y_test.shape)


data_train = pd.read_csv("real example-traindata.csv")
print(data_train.shape)
data_train = data_train.iloc[0:data_train.shape[0], 1:]
y_train = np.array(data_train)
print(y_train.shape)
R = np.array(data_train.Confirmedcases)
print(R.shape)


All = np.r_[y_test, y_train]
robust = preprocessing.RobustScaler()
All = robust.fit_transform(All)
y_test = All[:y_test.shape[0], :]
y_train = All[y_test.shape[0]:, :]
print(y_test.shape)
print(y_train.shape)


S0 = 100
q1 = np.quantile(R, 0.35)
q3 = np.quantile(R, 0.50)
q1_ind = np.where(R >= q1)[0][:]
q3_ind = np.where(R <= q3)[0][:]
IQR_ind = np.intersect1d(q1_ind, q3_ind)
# print(IQR_ind)
y_IQR = y_train[IQR_ind, :]
print("y_IQR.shape:", y_IQR.shape)
row_rand_array = np.arange(y_IQR.shape[0])
np.random.shuffle(row_rand_array)
# print(row_rand_array)
extract = row_rand_array[0:S0]
y_train = y_train[extract, :]
print(y_train.shape)


index = []
for i in range(S0):
    index.append(i)
bootstrap_index = np.random.choice(index, 1000, replace=True, p=None)
y_train_boot = y_train[bootstrap_index, :]
print("y_train_boot.shape:", y_train_boot.shape)


S0 = 100
y_S0 = y_train[:S0, :]
print("y_S0.shape:", y_S0.shape)


X = y_test


Movewindow = 5


y_n0 = -1*np.ones(y_S0.shape[0])
y_nw = np.ones(Movewindow)
y_final = np.r_[y_n0, y_nw]
print("y_final.shape:", y_final.shape)

# 取参考样本后X位形成X+1位移动窗口(96，97，98，99)
X_w = np.r_[y_S0[y_S0.shape[0]-Movewindow+1:, :], X]
print("X_w.shape:", X_w.shape)


if X_w.shape[0] >= Movewindow:
    Movewindow_X = X_w[:Movewindow, :]
else:
    print("The sample size is smaller than the capacity of the moving window.")

# 启动SVM算法
svm_model = svm.SVC(C=0.7, kernel="rbf", degree=3, gamma=0.4,
              coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
              class_weight=None, verbose=False, max_iter=-1, decision_function_shape="ovo",
              random_state=None)


a = 0.05
cl_S0 = []
cl_1 = []
y_move =y_train_boot[S0:, :]

for j in range(100):
    y_move = y_move[np.random.choice(y_move.shape[0], y_move.shape[0]), :]
    for i in range(y_move.shape[0]-Movewindow+1):
        y_w_move = y_move[i:Movewindow+i, :]
        y_w_final = np.r_[y_S0, y_w_move]
        svm_model.fit(y_w_final, y_final)
        decision_train = svm_model.decision_function(y_w_move)
        g_nw_train = 1 / (np.exp(-1 * decision_train) + 1)
        D_SVM_train = np.mean(g_nw_train)
        cl_S0.append(D_SVM_train)
    cl = np.quantile(cl_S0, 1-a)
    cl_1.append(cl)
print("len(cl_1):", len(cl_1))
cl_final = np.mean(cl_1)
radius = cl_final
print("CL:", radius)


D_SVM_f = []
n_alarm = 0
for i in range(X_w.shape[0]-Movewindow+1):
    New_X = np.array(X_w[Movewindow+i-1:Movewindow+i, :])
    New_X_w = X_w[i:Movewindow+i, :]
    y_combine = np.r_[y_S0, New_X_w]
    svm_model.fit(y_combine, y_final)
    decision_nw = svm_model.decision_function(New_X_w)
    g_nw = 1/(np.exp(-1*decision_nw)+1)
    D_SVM_new = np.mean(g_nw)
    print("Statistics:", D_SVM_new)

    if D_SVM_new > radius:
        D_SVM_f.append(D_SVM_new)
        n_alarm = n_alarm + 1
    elif D_SVM_new <= radius:
        D_SVM_f.append(D_SVM_new)
    else:
        print("Error")

D_SVM_f = np.array([D_SVM_f]).T
n_alarm_indicate = np.where(D_SVM_f > radius)[0][:]
print("n_alarm_indicate:", n_alarm_indicate)


def plot_distance(radius, distance):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    x_ticks_list = [0, 49, 99, 149, 199, 249, 299]
    x_ticks_real = [1, 50, 100, 150, 200, 250, 300]
    x_ticks_date = ['1-Mar', '20-Apr', '9-Jun', '29-Jul', '17-Sept', '6-Nov', '26-Dec']
    xticks(x_ticks_list, x_ticks_date, fontsize=12)
    yticks(fontsize=12)

    ax.plot(radius,
            color='r',
            linestyle='-',
            marker='None',
            linewidth=3,
            markeredgecolor='k',
            markerfacecolor='w',
            markersize=6,
            zorder=2)

    ax.plot(distance,
            color='k',
            linestyle=':',
            marker='o',
            linewidth=1,
            markeredgecolor='k',
            markerfacecolor='C4',
            markersize=6,
            zorder=1)

    ax.set_xlabel('Time (days)', fontsize=16)
    ax.set_ylabel('Statistics', fontsize=16)

    ax.legend(["D-SVM Control Limit"],
                ncol=1, loc=4,
                edgecolor='black',
                markerscale=1, fancybox=True)
    ax.yaxis.grid()
    plt.show()

n = D_SVM_f.shape[0]
radius = np.ones((n, 1))*radius
plot_distance(radius, D_SVM_f)
print("n_alarm:", n_alarm)

