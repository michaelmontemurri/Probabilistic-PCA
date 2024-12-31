import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import eigh
from algos import MixtureOfPPCA  

def cluster_demo():
    # Parameters
    n_samples = 700
    n_features = 3
    n_components = 5
    n_latent_dims = 2
    
    # Generate synthetic 3D data
    def generate_random_covariance(n_features):
        A = np.random.rand(n_features, n_features)
        return np.dot(A, A.T)

    # Make random cluster means and covariances
    cluster_means = [np.random.rand(n_features) * 10 for _ in range(n_components)]
    cluster_covars = [generate_random_covariance(n_features) for _ in range(n_components)]

    # Stack data from each cluster
    X = np.vstack([
        np.random.multivariate_normal(cluster_means[k], cluster_covars[k], size=n_samples // n_components)
        for k in range(n_components)
    ])

    #  Fit MPPCA model
    model = MixtureOfPPCA(n_components=n_components, q=n_latent_dims, max_iter=100, tol=1e-4)
    model.fit(X)

    # Posterior responsibilities for each point
    responsibilities = model.predict(X)
    labels = np.argmax(responsibilities, axis=1)

    # Fit classical PCA
    pca = PCA(n_components=n_latent_dims)
    pca.fit(X)

    # Reconstruction MSE 
    # MPPCA reconstruction (each point is assigned to the component of highest responsibility)
    X_recon_mppca = model.reconstruct(X)
    mse_mppca = np.mean((X - X_recon_mppca) ** 2)

    # PCA reconstruction
    X_proj_pca = pca.transform(X)
    X_recon_pca = pca.inverse_transform(X_proj_pca)
    mse_pca = np.mean((X - X_recon_pca) ** 2)

    print("\n===== Reconstruction MSE =====")
    print(f"PCA (global subspace): {mse_pca:.4f}")
    print(f"MPPCA (local subspaces): {mse_mppca:.4f}")

    # Visualization
    # Visualize 3D scatter plus the local subspaces from MPPCA and the global subspace from PCA.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.1, label="Data")

    # Plot each component's local principal directions
    # We have means_[k], W_[k], sigma2_[k]. 
    # The local covariance is: Sigma_k = W_k W_k^T + sigma2_k I.
    for k in range(n_components):
        mean_k = model.means_[k]
        W_k = model.W_[k]          # (d, q)
        sig2_k = model.sigma2_[k]
        # local covariance
        Sigma_k = W_k @ W_k.T + sig2_k * np.eye(n_features)

        # eigen-decomposition
        eigvals, eigvecs = eigh(Sigma_k)
        # top eigenvectors = last n_latent_dims columns
        top_eigvecs = eigvecs[:, -n_latent_dims:]

        # plot them as arrows from mean_k
        for j in range(n_latent_dims):
            vec = top_eigvecs[:, j]
            ax.quiver(*mean_k, *vec, color=f'C{k}', alpha=0.6, linewidth=1)

    # Plot the global principal directions from PCA
    global_mean = np.mean(X, axis=0)
    # pca.components_ shape: (n_latent_dims, n_features)
    # each row is a principal direction
    for j in range(n_latent_dims):
        ax.quiver(*global_mean, *pca.components_[j], color='k', alpha=0.8, linewidth=2, label=f"Global PC{j+1}")

    ax.set_title("MPPCA vs PCA Subspaces")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    cluster_demo()