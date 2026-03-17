class SIDARTHE:
    def __init__(self, T, N, I, D, A, R, T1, H, E, r1, r2, r3, r4, r5, b1, b2, b3, b4, a1, a2, g1, g2, u1, u2, t):
        self.N = N
        self.I = [I]
        self.D = [D]
        self.A = [A]
        self.R = [R]
        self.T1 = [T1]
        self.H = [H]
        self.E = [E]
        self.S = [N-I-D-A-R-T1-H-E]
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.r5 = r5
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.a1 = a1
        self.a2 = a2
        self.g1 = g1
        self.g2 = g2
        self.u1 = u1
        self.u2 = u2
        self.t = t
        self.T = T

    def calc(self):
        if len(self.T) == len(self.S):
            return
        for i in range(0, len(self.T) - 1):
            if self.S[i] < self.b1*self.I[i]+self.b2*self.D[i]+self.b3*self.A[i]+self.b4*self.R[i]:
                self.S.append(0)
                self.I.append(self.I[i] + self.S[i] - (self.a1 + self.g1 + self.r1) * self.I[i])
                self.D.append(self.D[i] + self.a1 * self.I[i] - (self.g2 + self.r2) * self.D[i])
                self.A.append(self.A[i] + self.g1 * self.I[i] - (self.r3 + self.u1 + self.a2) * self.A[i])
                self.R.append(self.R[i] + self.g2 * self.D[i] + self.a2 * self.A[i] - (self.r4 + self.u2) * self.R[i])
                self.T1.append(self.T1[i] + self.u1 * self.A[i] + self.u2 * self.R[i] - (self.r5 + self.t) * self.T1[i])
                self.H.append(self.H[i] + self.r1 * self.I[i] + self.r2 * self.D[i] + self.r3 * self.A[i] + self.r4 * self.R[i] + self.r5 * self.T1[i])
                self.E.append(self.E[i] + self.t * self.T1[i])
            else:
                self.S.append(self.S[i]-self.b1*self.I[i]-self.b2*self.D[i]-self.b3*self.A[i]-self.b4*self.R[i])
                self.I.append(self.I[i]+self.b1*self.I[i]+self.b2*self.D[i]+self.b3*self.A[i]+self.b4*self.R[i]-(self.a1+self.g1+self.r1)*self.I[i])
                self.D.append(self.D[i]+self.a1*self.I[i]-(self.g2+self.r2)*self.D[i])
                self.A.append(self.A[i]+self.g1*self.I[i]-(self.r3+self.u1+self.a2)*self.A[i])
                self.R.append(self.R[i]+self.g2*self.D[i]+self.a2*self.A[i]-(self.r4+self.u2)*self.R[i])
                self.T1.append(self.T1[i]+self.u1*self.A[i]+self.u2*self.R[i]-(self.r5+self.t)*self.T1[i])
                self.H.append(self.H[i]+self.r1*self.I[i]+self.r2*self.D[i]+self.r3*self.A[i]+self.r4*self.R[i]+self.r5*self.T1[i])
                self.E.append(self.E[i]+self.t*self.T1[i])


    def plot(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False

        if len(self.S) == 1:
            self.calc()
        plt.figure()
        plt.plot(self.T, self.S, color='b', label='Susceptible')
        plt.plot(self.T, self.I, color='r', label='Infected')
        plt.plot(self.T, self.D, color='y', label='Diagnosed')
        plt.plot(self.T, self.A, color='c', label='Ailing')
        plt.plot(self.T, self.R, color='m', label='Recognized')
        plt.plot(self.T, self.T1, color='g', label='Threatened')
        plt.plot(self.T, self.H, color='k', label='Healed')
        plt.plot(self.T, self.E, color='xkcd:orange', label='Extinct')
        plt.grid(False)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Proportion of population")
        plt.show()


if __name__ == "__main__":
    T = [i for i in range(150)]
    s = SIDARTHE(T, 1, 0.0005, 0, 0, 0, 0, 0, 0, 0.0499, 0.0499, 0.1480, 0.1480, 0.1480, 0.1779, 0.0048, 0.0048, 0.3926, 0.1238, 0.1799, 0.043, 0.043, 0.0255, 0.0336, 0.0127)
    s.plot()