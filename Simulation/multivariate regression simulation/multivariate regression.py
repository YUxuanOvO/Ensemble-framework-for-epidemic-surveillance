import numpy as np
import math


def static():
    A = np.array([[-0.2310, -0.0816, -0.2662], [-0.3241, 0.7055, -0.2158],
                  [-0.217, -0.3056, -0.5207], [-0.4089, -0.3442, -0.4501],
                  [-0.6408, 0.3102, 0.2372], [-0.4655, -0.433, 0.5938]])
    t1 = np.random.normal(0, 1)
    t2 = np.random.normal(0, 0.8)
    t3 = np.random.normal(0, 0.6)
    t = np.array([[t1, t2, t3]]).T
    e = np.random.normal(0, 0.2, [6, 1])
    xk = np.dot(A, t) + e
    return xk

def time_Varying(i):
    A = np.array([[-0.2310, -0.0816, -0.2662], [-0.3241, 0.7055, -0.2158],
                  [-0.217, -0.3056, -0.5207], [-0.4089, -0.3442, -0.4501],
                  [-0.6408, 0.3102, 0.2372], [-0.4655, -0.433, 0.5938]])
    t1 = np.random.normal(0, 1)
    t2 = np.random.normal(0, 0.8)
    t3 = np.random.normal(0, 0.6)
    t = np.array([[t1, t2, t3]]).T
    e = np.random.normal(0, 0.2, [6, 1])
    timevary = np.array([[0.005*(i-300), 0, -0.004*(i-300), 0, 0, 0.007*(i-300)]]).T
    xk = np.dot(A, t) + e + timevary
    return xk

def time_Varying_fault(i):
    A= np.array([[-0.2310, -0.0816, -0.2662], [-0.3241, 0.7055, -0.2158],
                [-0.217, -0.3056, -0.5207], [-0.4089, -0.3442, -0.4501],
                [-0.6408, 0.3102, 0.2372], [-0.4655, -0.433, 0.5938]])
    t1 = np.random.normal(0, 1)
    t2 = np.random.normal(0, 0.8)
    t3 = np.random.normal(0, 0.6)
    t = np.array([[t1, t2, t3]]).T
    e = np.random.normal(0, 0.2, [6, 1])
    timevary = np.array([[0.005*(i-300), 0, -0.004*(i-300), 0, 0, 0.007*(i-300)]]).T
    xk = np.dot(A, t) + e + timevary
    fault = np.array([[1, 1, 1, 1, 1, 1]]).T
    xk_fault = xk + 6*fault
    return xk_fault


n = 1000
xk_final = np.array([np.zeros(6)]).T

for i in range(1, n+1):
    if i <= 300:
        xk = static()
        xk_final = np.c_[xk_final, xk]
    elif i>300 and i<=800:
        xk = time_Varying(i)
        xk_final = np.c_[xk_final, xk]
    else:
        xk = time_Varying_fault(i)
        xk_final = np.c_[xk_final, xk]

xk_final = np.delete(xk_final, 0, axis=1).T
print(xk_final)
np.savetxt('Y_f_off1dynamics.csv', xk_final, delimiter=',')


n = 1000
xk_final_off = np.array([np.zeros(6)]).T

for i in range(1, n+1):
    xk_off = static()
    xk_final_off = np.c_[xk_final_off, xk_off]
xk_final_off = np.delete(xk_final_off, 0, axis=1).T
print(xk_final_off)
np.savetxt('Y_f_off1static.csv', xk_final_off, delimiter=',')