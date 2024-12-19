import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from algos import ProbabilisticKernelPCA
from sklearn.decomposition import KernelPCA

# generate synthetic concentric circles dataset
def generate_data():
    X, y = make_circles(n_samples=500, factor=0.3, noise=0.2)  # two circles with added noise
    return X, y

# plot the data in 2d space
def plot_data(X, y, title="Original Data"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=20, alpha=0.8)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# plot data in latent (transformed) space
def plot_latent_space(Z, y, title="Latent Space"):
    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=20, alpha=0.8)
    plt.title(title)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()

if __name__ == "__main__":
    # generate and visualize the original dataset
    X, y = generate_data()
    plot_data(X, y, title="Original Data")

    # apply probabilistic kernel pca using the ml method
    print("applying probabilistic kernel pca with ml...")
    pkpca_ml = ProbabilisticKernelPCA(n_components=2, kernel="rbf", gamma=10, method="ML")
    pkpca_ml.fit(X)  # fit the model to the data
    Z_pkpca_ml = pkpca_ml.latent_vars  # get the transformed data
    plot_latent_space(Z_pkpca_ml, y, title="Probabilistic Kernel PCA (ML) Latent Space")

    # apply probabilistic kernel pca using the em method
    print("applying probabilistic kernel pca with em...")
    pkpca_em = ProbabilisticKernelPCA(n_components=2, kernel="rbf", gamma=10, max_iter=100, tol=1e-4, method="EM")
    pkpca_em.fit(X)  # fit the model to the data
    Z_pkpca_em = pkpca_em.latent_vars  # get the transformed data
    plot_latent_space(Z_pkpca_em, y, title="Probabilistic Kernel PCA (EM) Latent Space")

    # compare log-likelihoods between methods
    print("log-likelihood comparison:")
    print(f"log-likelihood (ml): {pkpca_ml.log_likelihoods[-1]}")
    print(f"log-likelihood (em): {pkpca_em.log_likelihoods[-1]}")
