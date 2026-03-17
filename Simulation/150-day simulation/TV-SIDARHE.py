import math

class adjust_SIDARTHE:
    def __init__(self, T, N, I, D, A, R, H, E,
                 α, β, ε, ζ, θ, ρ, σ, κ, λ, μ, a, b):
        self.N = N
        self.I = [I]
        self.D = [D]
        self.A = [A]
        self.R = [R]
        self.H = [H]
        self.E = [E]
        self.S = [N-I-D-A-R-H-E]
        self.α = α
        self.β = β
        self.ε = ε
        self.ζ = ζ
        self.θ = θ
        self.ρ = ρ
        self.σ = σ
        self.κ = κ
        self.λ = λ
        self.μ = μ
        self.a = a
        self.b = b
        self.T = T

    def calc(self):
        if len(self.T) == len(self.S):
            return
        for i in range(0, len(self.T) - 1):
            if self.S[i] < self.b**(-i)*self.α*self.I[i]-self.b**(-i)*self.β*self.A[i]:
                self.S.append(0)
                self.I.append(self.I[i] + self.S[i] - ((math.log(i+0.1**10, self.a)+1)*self.ε+self.ζ+self.ρ) * self.I[i])
                self.D.append(self.D[i] + (math.log(i+0.1**10, self.a)+1)*self.ε*self.I[i]-(self.ζ+self.λ)*self.D[i])
                self.A.append(self.A[i] + self.ζ*self.I[i]-((math.log(i+0.1**10, self.a)+1)*self.θ+self.σ)*self.A[i])
                self.R.append(self.R[i] + self.ζ*self.D[i]+(math.log(i+0.1**10, self.a)+1)*self.θ*self.A[i]-(self.κ+self.μ)*self.R[i])
                self.H.append(self.H[i] + self.ρ*self.I[i]+self.λ*self.D[i]+self.σ*self.A[i]+self.κ*self.R[i])
                self.E.append(self.E[i] + self.μ*self.R[i])
            else:
                self.S.append(self.S[i] - self.b**(-i)*self.α*self.I[i]-self.b**(-i)*self.β*self.A[i])
                self.I.append(self.I[i] + self.b**(-i)*self.α*self.I[i]+self.b**(-i)*self.β*self.A[i]-((math.log(i+0.1**10, self.a)+1)*self.ε+self.ζ+self.ρ)*self.I[i])
                self.D.append(self.D[i] + (math.log(i+0.1**10, self.a)+1)*self.ε*self.I[i]-(self.ζ+self.λ)*self.D[i])
                self.A.append(self.A[i] + self.ζ*self.I[i]-((math.log(i+0.1**10, self.a)+1)*self.θ+self.σ)*self.A[i])
                self.R.append(self.R[i] + self.ζ*self.D[i]+(math.log(i+0.1**10, self.a)+1)*self.θ*self.A[i]-(self.κ+self.μ)*self.R[i])
                self.H.append(self.H[i] + self.ρ*self.I[i]+self.λ*self.D[i]+self.σ*self.A[i]+self.κ*self.R[i])
                self.E.append(self.E[i] + self.μ*self.R[i])


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
        plt.plot(self.T, self.H, color='k', label='Healed')
        plt.plot(self.T, self.E, color='xkcd:orange', label='Extinct')
        plt.grid(False)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Proportion of population")
        plt.show()


if __name__ == "__main__":
    T = [i for i in range(150)]
    s = adjust_SIDARTHE(T, 1, 0.001, 0, 0.001, 0, 0, 0, 0.8, 0.8, 0.016, 0.07,
                        1.56e-02, 1.68e-02, 0.55, 0.023, 0.14, 0.1238, 2, 1.05)
    s.plot()
