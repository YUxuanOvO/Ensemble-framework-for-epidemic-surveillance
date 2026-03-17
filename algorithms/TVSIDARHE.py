import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import r2_score
from sklearn import preprocessing



start_time = time.time()


data = pd.read_csv("traindata.csv") # short/long-term
N = 140005e4
D = np.array(data.D)
R = np.array(data.R)
H = np.array(data.H)
E = np.array(data.E)
y = np.array([D, R, H, E])
y = np.transpose(y)


data_t = pd.read_csv("testdata.csv") # short/long-term
D_t = np.array(data_t.D)
R_t = np.array(data_t.R)
H_t = np.array(data_t.H)
E_t = np.array(data_t.E)
y_t = np.array([D_t, R_t, H_t, E_t])
y_t = np.transpose(y_t)
pre_long = y.shape[0]+y_t.shape[0]


def rks4(f, a, b, Za, M):
    # M + 1 steps in total
    h = (b - a) / M
    t = np.linspace(a, b, M + 1).reshape(M + 1, 1)
    Z = np.zeros((M + 1, Za.size))
    Z[0, :] = Za
    for i in range(1, M + 1):
        k1 = h * f(t[i - 1], np.transpose(Z[i - 1, :]))
        k2 = h * f(t[i - 1] + h / 2, np.transpose(Z[i - 1, :] + k1 / 2))
        k3 = h * f(t[i - 1] + h / 2, np.transpose(Z[i - 1, :] + k2 / 2))
        k4 = h * f(t[i - 1] + h, np.transpose(Z[i - 1, :] + k3))
        Z[i, :] = Z[i - 1, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return (t,Z)


def SIDARHE(t, y, arg, N):
    (α, β, ε, ζ, θ, ρ, σ, κ, λ, μ, a, b) = (arg[0], arg[1], arg[2], arg[3], arg[4], arg[5],
                                 arg[6], arg[7], arg[8], 0.1**10, arg[9], arg[10])
    dS = -(b**(-t)*α * y[1] + b**(-t)*β * y[2]) * y[0] / N
    dI = (b**(-t)*α * y[1] + b**(-t)*β * y[2]) * y[0] / N - ((math.log(t,a)+1)*ε + ζ + ρ) * y[1]
    dA = ζ * y[1] - ((math.log(t,a)+1)*θ + σ) * y[2]
    dD = (math.log(t,a)+1)*ε * y[1] - (ζ + λ) * y[3]
    dR = ζ * y[3] + (math.log(t, a)+1)*θ * y[2] - (κ + μ) * y[4]
    dH = ρ * y[1] + σ * y[2] + λ * y[3] + κ * y[4]
    dE = μ * y[4]
    return np.array([dS, dI, dA, dD, dR, dH, dE], dtype=object).T


# Initial value setting (The first value is consistent with the actual value)
S=N-y[0][0]-y[0][1]-y[0][2]-y[0][3]-20000-10000
y0 = np.array([S,20000,10000,y[0][0],y[0][1],y[0][2],y[0][3]])


def rfun(args, ydata, N):
    len = ydata.shape[0]
    _,yval = rks4(lambda t,y: SIDARHE(t,y, args, N), 1, len, y0, len-1)
    weights = np.array([0,1,0,0])
    return np.sum(np.abs(yval[:,3:] - ydata)*weights, axis=1)



def rfun1(arg):
    arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9], arg[10] = arg
    len = y.shape[0]
    _,yval = rks4(lambda t, y: SIDARHE(t, y, arg, N), 1, len, y0, len-1)
    weights = np.array([[0], [1], [0], [0]])
    return np.sum(np.dot(np.abs(yval[:, 3:] - y), weights))



from scipy.optimize import least_squares
arg0 = np.array([0.11,0.105,0.143,0.57,0.196,0.011,0.017,0.371,0.456,3,3])#初始参数设置(0.1961)
optiargs = least_squares(lambda arg: rfun(arg, y, N), arg0,bounds=((0,0,0,0,0,0,0,0,0,0,0),
(1,1,1,1,1,1,1,1,1,np.inf,np.inf)),max_nfev=10000)

print(optiargs)
len = y.shape[0]
tval, yval = rks4(lambda t,y: SIDARHE(t,y, optiargs.x, N), 1, pre_long, y0, pre_long-1)


sns.set()
fig, ax = plt.subplots()
t_True = np.linspace(1, y.shape[0], y.shape[0]).reshape(y.shape[0], 1)
plt.plot(tval, yval[:,4], 'r--', linewidth=1.5, label='Fit')
plt.plot(t_True, y[:,1], 'k--', linewidth=1.5, label='True')
plt.plot(tval[y.shape[0]-1:tval.shape[0]-1,:],
         y_t[:,1], 'y-', linewidth=2, label='Test')


plt.legend(loc='best')
plt.xlabel('days')
R2 = r2_score(yval[:y.shape[0],4], y[:,1])
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
err = mape(y_t[:,1],yval[y.shape[0]:,4])
plt.show()


y_predict = yval[y.shape[0]:,3:]
y_final = np.r_[y,y_predict]

y_mean = np.array([np.mean(y_final, axis=0)])
y_mean_final = y_mean
for i in range(y_final.shape[0]-1):
    y_mean_final = np.r_[y_mean_final,y_mean]

y_max = np.array([np.max(y_final, axis=0)])
y_max_final = y_max
for i in range(y_final.shape[0]-1):
    y_max_final = np.r_[y_max_final,y_max]

y_min = np.array([np.min(y_final, axis=0)])
y_min_final = y_min
for i in range(y_final.shape[0]-1):
    y_min_final = np.r_[y_min_final,y_min]

y_var = np.array([np.var(y_final, axis=0)])
y_std = np.sqrt(y_var * y_final.shape[0]/(y_final.shape[0]-1))
y_std_final = y_std
for i in range(y_final.shape[0]-1):
    y_std_final = np.r_[y_std_final,y_std]


y_final1 = (y_final-y_mean_final)/(y_max_final-y_min_final)
y_final2= (y_final-y_mean_final)/y_std_final
robust1 = preprocessing.RobustScaler()
y_final3 = robust1.fit_transform(y_final)

np.savetxt('y_final.csv', y_predict, delimiter = ',')

end_time = time.time()
running_time = end_time - start_time
print(running_time)

