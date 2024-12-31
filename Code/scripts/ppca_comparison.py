import numpy as np
import time
import matplotlib.pyplot as plt
from algos import PPCAEigen, PPCAEM

def ppca_comparison():
    """
    Compare computation times for PPCA (Eigen) vs. PPCA (EM) on large synthetic data.
    We'll vary d (the dimensionality) while keeping N fixed, or vice versa.
    """
    np.random.seed(0)
    
    N = 500
    dimensions = [50, 200, 400]

    q = 5  # latent dimension
    max_iter = 30  # for demonstration

    times_eigen = []
    times_em = []

    for d in dimensions:
        # Generate random data matrix X, shape (N, d)
        X = np.random.randn(N, d)  # standard normal

        # PPCA (Eigen)
        start = time.time()
        ppca_eig = PPCAEigen(q=q, max_iter=max_iter, tol=1e-5, random_state=42)
        ppca_eig.fit(X)
        end = time.time()
        t_eigen = end - start
        times_eigen.append(t_eigen)

        # PPCA (EM)
        start = time.time()
        ppca_em = PPCAEM(q=q, max_iter=max_iter, tol=1e-5, random_state=42)
        ppca_em.fit(X)
        end = time.time()
        t_em = end - start
        times_em.append(t_em)

        print(f"d={d}, PPCA(Eigen) time={t_eigen:.3f}s, PPCA(EM) time={t_em:.3f}s")

    # Plot the time vs. dimension
    plt.figure(figsize=(6,4))
    plt.plot(dimensions, times_eigen, 'o--', label='PPCA (Eigen)')
    plt.plot(dimensions, times_em, 's--', label='PPCA (EM)')
    plt.xlabel("Dimension d")
    plt.ylabel("Time (seconds)")
    plt.title("Computation Time vs. Dimension")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ppca_comparison()
