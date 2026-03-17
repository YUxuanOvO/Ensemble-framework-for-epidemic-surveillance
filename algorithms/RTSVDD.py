import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from src.BaseSVDD import BaseSVDD
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from sklearn import preprocessing

import time
start_time = time.time()


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


num_trains = y_train.shape[0]


sample_ids = np.arange(0,num_trains)
print(len(sample_ids))

svdd = BaseSVDD(C=1, gamma=0.4, kernel='rbf', display='off')
svdd.fit(y_train)
distances = svdd.get_distance(y_train)
print(len(distances))

distance_dict = dict(zip(sample_ids,distances))



sorted_items = sorted(distance_dict.items(), key=lambda x: x[1])


n = len(sorted_items)
q1_index = int(n * 0.35)
q3_index = int(n * 0.50)


selected_sample_ids = [item[0] for item in sorted_items[q1_index:q3_index]]

print(len(selected_sample_ids))

valid_indices = [idx for idx in selected_sample_ids if 0 <= idx < y_train.shape[0]]

y_train_pro = y_train[valid_indices, :]
print(len(y_train_pro))


S0 = 100
S0_0 = y_train_pro.shape[0]
row_indices = np.random.choice(S0_0, size=S0, replace=False)

y_S0 = y_train[row_indices, :]

X = y_test

Movewindow = 5


X_w = np.r_[y_S0[y_S0.shape[0]-Movewindow+1:, :], X]


if X_w.shape[0] >= Movewindow:
    Movewindow_X = X_w[:Movewindow, :]
else:
    print("The sample size is smaller than the capacity of the moving window.")


b = 0.1
weight_f = np.array([[1/(b*(Movewindow-1)+1)]])
for i in range(Movewindow-1):
    weight_t = np.array([[1/(b*(Movewindow-i-2)+1)]])
    weight_f = np.r_[weight_f, weight_t]
print("weight_f", weight_f)
weight_nol = weight_f/(np.sum(weight_f))
print("weight_nol", weight_nol)


b = 0
weight_f_S0 = np.array([[1/(b*(y_S0.shape[0]+Movewindow-1)+1)]])
for i in range(y_S0.shape[0]+Movewindow-1):
    weight_t_S0 = np.array([[1/(b*(y_S0.shape[0]+Movewindow-i-2)+1)]])
    weight_f_S0 = np.r_[weight_f_S0, weight_t_S0]
print("weight_f_S0", weight_f_S0)


TSVDD_f = []
radius_f = []
n_alarm = 0

# svdd object using rbf kernel(S0)

a = 0.05
svdd = BaseSVDD(C=0.05, gamma=0.8, kernel='rbf', display='off')


for i in range(X_w.shape[0]-Movewindow+1):
    New_X = np.array(X_w[Movewindow+i-1:Movewindow+i, :])
    New_X_w = X_w[i:Movewindow+i, :]
    y_combine = np.r_[y_S0, New_X_w]

    svdd.C = 0.05
    svdd.fit(y_combine, weight=weight_f_S0)
    distance_S0 = svdd.get_distance(y_S0).A.flatten()
    radius_S0 = svdd.radius
    R_S0 = np.quantile(distance_S0, 1-a)
    print("radius_S0", radius_S0)
    print("R_S0", R_S0)
    radius_final = R_S0

    TSVDD_new = np.dot(weight_nol.T, svdd.get_distance(New_X_w))[0, 0]
    print("TSVDD_new", TSVDD_new)
    if TSVDD_new > radius_final:
        radius_f.append(radius_final)
        TSVDD_f.append(TSVDD_new)
        n_alarm = n_alarm + 1
    elif TSVDD_new <= radius_final:
        radius_f.append(radius_final)
        TSVDD_f.append(TSVDD_new)
        y_S0 = np.r_[y_S0, New_X]
        y_S0 = np.delete(y_S0, 0, axis=0)
    else:
        print("Error")

radius_f = np.array([radius_f]).T
TSVDD_f = np.array([TSVDD_f]).T
n_alarm_indicate = np.where(TSVDD_f > radius_f)[0][:]
print("n_alarm_indicate", n_alarm_indicate)


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

    ax.legend(["RTSVDD Control Limit"],
                ncol=1, loc=4,
                edgecolor='black',
                markerscale=1, fancybox=True)
    ax.yaxis.grid()
    plt.show()

plot_distance(radius_f, TSVDD_f)
print("n_alarm", n_alarm)

