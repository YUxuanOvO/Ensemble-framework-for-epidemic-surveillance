import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from src.BaseSVDD import BaseSVDD
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

y_IQR = y_train[IQR_ind, :]
print("y_IQR.shape:", y_IQR.shape)
row_rand_array = np.arange(y_IQR.shape[0])
np.random.shuffle(row_rand_array)
extract = row_rand_array[0:S0]

extract.sort()

extract = abs(np.sort(-row_rand_array[0:S0]))
y_train = y_train[extract, :]
print(y_train.shape)


S0 = 100
y_S0 = y_train[:S0, :]
print("y_S0.shape:", y_S0.shape)

X = y_test


b = 0.4
def timefactor(b, y_S0):
    weight_f_S0 = np.array([[1/(b*(y_S0.shape[0]-1)+1)]])
    for i in range(y_S0.shape[0]-1):
        weight_t_S0 = np.array([[1/(b*(y_S0.shape[0]-i-2)+1)]])
        weight_f_S0 = np.r_[weight_f_S0, weight_t_S0]
    return weight_f_S0


weight_S0 = timefactor(b, y_S0)


svdd = BaseSVDD(C=0.8, gamma=0.07, kernel='rbf', display='off')
svdd.fit(y_S0, weight=weight_S0)
distance_S0 = svdd.get_distance(y_S0)

distance_S0 = distance_S0.A.flatten()

a = 0.05
d = 0.05
cl_S01 = []
ul_S01 = []

for i in range(10000):
    bootstrap_S01 = np.random.choice(distance_S0, len(distance_S0), replace=True, p=None)
    cl_boot1 = np.quantile(bootstrap_S01, 1-a)
    cl_S01.append(cl_boot1)
    ul_boot1 = np.quantile(bootstrap_S01, 1-a-d)
    ul_S01.append(ul_boot1)
cl_S01 = np.mean(cl_S01)
ul_S01 = np.mean(ul_S01)
print("cl_S01:", cl_S01)
print("ul_S01:", ul_S01)

cl_S02 = []
ul_S02 = []

for i in range(1000):

    bootstrap_S02 = y_S0[np.random.choice(y_S0.shape[0], y_S0.shape[0]), :]
    svdd.C = 0.5
    svdd.fit(bootstrap_S02, weight=weight_S0)
    distance2 = svdd.get_distance(bootstrap_S02).A.flatten()

    cl_boot2 = np.quantile(distance2, 1-a)
    cl_S02.append(cl_boot2)
    ul_boot2 = np.quantile(distance2, 1-a-d)
    ul_S02.append(ul_boot2)
cl_S02 = np.mean(cl_S02)
ul_S02 = np.mean(ul_S02)
print("cl_S02:", cl_S02)
print("ul_S02:", ul_S02)


cl_S0 = cl_S02
ul_S0 = ul_S02


distance_f = []
n_alarm = 0
for i in range(X.shape[0]):
    New_X = np.array(X[i:i+1, :])
    distance_New = np.mean(svdd.get_distance(New_X))

    if distance_New > cl_S0:
        distance_f.append(distance_New)
        n_alarm = n_alarm + 1
    elif distance_New <= ul_S0:
        distance_f.append(distance_New)
        y_S0 = np.r_[y_S0, New_X]
    elif distance_New > ul_S0 and distance_New <= cl_S0:
        distance_f.append(distance_New)
        y_S0 = np.r_[y_S0, New_X]
        weight_S0 = timefactor(b, y_S0)
        svdd.C = 0.5
        svdd.fit(y_S0, weight=weight_S0)
    else:
        print("Error")



distance_f = np.array([distance_f]).T
n_alarm_indicate = np.where(distance_f > cl_S0)[0][:]
print("n_alarm_indicate：", n_alarm_indicate)


n = distance_f.shape[0]
radius_cl = np.ones((n, 1))*cl_S0
radius_ul = np.ones((n, 1))*ul_S0


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

    ax.legend(["Time-adaptive SVDD Control Limit"],
                ncol=1, loc=4,
                edgecolor='black',
                markerscale=1, fancybox=True)
    ax.yaxis.grid()
    plt.show()

plot_distance(radius_cl, distance_f)
print("n_alarm:", n_alarm)

