import numpy as np
import matplotlib.pyplot as plt
def map_feature(X1, X2, degree=6):
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)

    out = []

    for i in range(1, degree+1):
        for j in range(i+1):
            out.append((X1**(i-j)) * (X2**j))

    out = np.array(out)

    if X1.size == 1:       
        return out.flatten()  # returns (27,)
    else:
        return out.T         # returns (m,27)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_decision_boundary(w, b, X, y):

    if X.shape[1] <= 2:
        # Linear case
        plot_x = np.array([X[:,0].min(), X[:,0].max()])
        plot_y = (-1./w[1]) * (w[0]*plot_x + b)
        plt.plot(plot_x, plot_y, c="g")
    else:
        # Nonlinear / polynomial
        u = np.linspace(X[:,0].min()-0.1, X[:,0].max()+0.1, 50)
        v = np.linspace(X[:,1].min()-0.1, X[:,1].max()+0.1, 50)
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                features = map_feature(u[i], v[j])  # 1D array
                z[i,j] = float(np.dot(features, w) + b)  # convert to scalar

        z = z.T  # transpose for correct orientation
        plt.contour(u, v, z, levels=[0], colors='g')  # decision boundary
