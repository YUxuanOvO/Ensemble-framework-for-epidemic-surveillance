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


svdd = BaseSVDD(C=0.4, gamma=0.03, kernel='rbf', display='off')
svdd.fit(y_S0)


radius = svdd.radius
distance = svdd.get_distance(X)
n = distance.shape[0]
radius_f = np.ones((n, 1))*radius
print("radius:", radius)

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

    ax.legend(["K-SVDD Control Limit"],
                ncol=1, loc=4,
                edgecolor='black',
                markerscale=1, fancybox=True)
    ax.yaxis.grid()
    plt.show()

plot_distance(radius_f, distance)


n_alarm = 0
for i in range(n):
    if radius < distance[i][0]:
        n_alarm = n_alarm + 1
print("n_alarm:", n_alarm)


n_alarm_indicate = np.where(distance > radius_f)[0][:]
print("n_alarm_indicate:", n_alarm_indicate)



import time
end_time = time.time()
running_time = end_time - start_time
print("running_time:", running_time)