import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import r2_score

start_time = time.time()

data = pd.read_csv("traindata.csv") # short/long-term
N = 140005e4
D = np.array(data.D)
R = np.array(data.R)
H = np.array(data.H)
# Th = np.array(data.Th)
E = np.array(data.E)
y = np.array([D, R, H, E])
y = np.transpose(y)


data_t = pd.read_csv("testdata.csv") # short/long-term
D_t = np.array(data_t.D)
R_t = np.array(data_t.R)
H_t = np.array(data_t.H)
# Th_t = np.array(data_t.Th)
E_t = np.array(data_t.E)
y_t = np.array([D_t, R_t, H_t, E_t])
y_t = np.transpose(y_t)
pre_long=y.shape[0]+y_t.shape[0]



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

def prototype(t,y, arg, N):
    (α, β, γ, δ, ε, ζ, η, θ, μ, ν, τ, λ, κ, σ, ξ, ρ) = \
        (arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8],
        arg[9], 0.1 ** 10, arg[10], arg[11], arg[12], arg[13], arg[14])
    dS = -(α*y[1]+γ*y[2]+β*y[3]+δ*y[4])*y[0]/N
    dI = (α*y[1]+γ*y[2]+β*y[3]+δ*y[4])*y[0]/N-(ε+ζ+λ)*y[1]
    dA = ζ*y[1]-(θ+μ+κ)*y[2]
    dD = ε*y[1]-(η+ρ)*y[3]
    dR = η*y[3]+θ*y[2]-(ν+ξ)*y[4]
    dH = λ*y[1]+κ*y[2]+ρ*y[3]+ξ*y[4]+σ*y[6]
    dTh = μ*y[2]+ν*y[4]-(σ+τ)*y[6]
    dE = τ*y[6]
    return np.array([dS, dI, dA, dD, dR, dH, dTh, dE], dtype=object).T



S=N-y[0][0]-y[0][1]-y[0][2]-y[0][3]-20000-10000-100
y0 = np.array([S,20000,10000,y[0][0],y[0][1],y[0][2],100,y[0][3]])



def rfun(args, ydata, N):
    len = ydata.shape[0]
    _,yval = rks4(lambda t,y: prototype(t,y, args, N), 1, len, y0, len-1)
    weights = np.array([0,1])
    return np.sum(np.abs(yval[:,3:5] - ydata[:,0:2])*weights, axis=1)


def rfun1(arg):
    arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], \
    arg[9], arg[10] , arg[11], arg[12], arg[13], arg[14],arg[15],arg[16] = arg
    len = y.shape[0]
    _,yval = rks4(lambda t,y: prototype(t,y, arg, N), 1, len, y0, len-1)
    weights = np.array([[1],[1],[0],[0],[0]])
    return np.sum(np.dot(np.abs(yval[:,3:] - y),weights))



from scipy.optimize import least_squares
arg0 = np.array([0.1779,0.0048,0.0048,0.3926,0.1238,0.0430,0.0430,0.1799,
                 0.0499,0.1480,0.1480,0.0499,0.1480,0.0255,0.0336])
optiargs = least_squares(lambda arg: rfun(arg, y, N), arg0,
                         bounds=((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                                 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)),max_nfev=10000)

print(optiargs)
len = y.shape[0]
tval, yval = rks4(lambda t,y: prototype(t,y, optiargs.x, N), 1, pre_long, y0, pre_long-1)


sns.set()
fig, ax = plt.subplots()
t_True = np.linspace(1, y.shape[0], y.shape[0]).reshape(y.shape[0], 1)
plt.plot(tval, yval[:,4], 'r--', linewidth=1.5, label='Fit')
plt.plot(t_True, y[:,1], 'k--', linewidth=1.5, label='True')
plt.plot(tval[y.shape[0]-1:tval.shape[0]-1,:],
         y_t[:,1], 'y-', linewidth=2, label='Test')


plt.legend(loc='best')
plt.xlabel('days')
#

R2 = r2_score(yval[:y.shape[0],4], y[:,1])
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
err = mape(y_t[:,1],yval[y.shape[0]:,4])
print(f"R-Square：{R2:.{3}}")
print(f"MAPE：{err:.{3}}")

plt.show()


end_time = time.time()
running_time = end_time - start_time
print(running_time)
