import pennylane as qml
from pennylane import numpy as np
from qbc_ipe import qbc_ipe_algorithm
import scipy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N = 2**4
    M = 2
    lambda_exact = np.array([-1.0, 2.0])

    def f(x):
        return np.dot(lambda_exact, x) + np.random.normal(0, 0.1)

    X = np.ones((N, M))
    X[:, 1] = np.random.random(N)
    y = np.array([f(x) for x in X])

    X_mp = la.inv(X.T @ X) @ X.T
    lambda_numerics = X_mp @ y

    lambda_approx = np.zeros(M)
    print("X_mp = ", X_mp)
    for i in range(M):
        a = X_mp[i, :].T
        rho, _, _, _, _, _ = qbc_ipe_algorithm(y, a, num_t_wires=10)
        lambda_approx[i] = rho

    print("lambda_exact = ", lambda_exact)
    print("lambda_approx = ", lambda_approx)
    print("X_mp @ y = ", X_mp @ y)

    ax = sns.scatterplot(x=X[:, 1], y=y, color="black", alpha=0.8)
    ax.set_xlabel("$\mathbf{x}$")
    ax.set_ylabel("$f(\mathbf{x}, \mathbf{\lambda})$")
    ax.plot(X[:, 1], X @ lambda_exact, label="Exact", alpha=0.8)
    ax.plot(X[:, 1], X @ lambda_approx, label="IPE", alpha=0.8)
    ax.plot(X[:, 1], X @ lambda_numerics, label="Numerics", alpha=0.8)
    ax.legend()

    plt.savefig("./plots/lin_reg_qbc_ipe_lambda_-1_2.png", dpi=300)
