import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm  # for color mapping
from sklearn.decomposition import PCA
from numpy.linalg import eigh


class MixtureOfPPCA:
    def __init__(self, n_components, n_latent_dims, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.n_latent_dims = n_latent_dims
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covars = None
        self.latent_covars = None

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covars = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.latent_covars = np.array([
            np.eye(self.n_latent_dims) for _ in range(self.n_components)
        ])

    def _e_step(self, X):
        n_samples, n_features = X.shape
        log_resps = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            diff = X - self.means[k]
            covar_k = self.covars[k]
            try:
                log_det_covar_k = np.log(np.linalg.det(covar_k) + 1e-6)
                inv_covar_k = np.linalg.inv(covar_k + np.eye(n_features) * 1e-6)
            except np.linalg.LinAlgError:
                log_det_covar_k = np.inf
                inv_covar_k = np.eye(n_features)
            log_resps[:, k] = (
                -0.5 * (np.sum(diff @ inv_covar_k * diff, axis=1) + log_det_covar_k)
                + np.log(self.weights[k] + 1e-6)
            )

        max_log_resps = np.max(log_resps, axis=1, keepdims=True)
        log_resps -= max_log_resps  # for numerical stability
        log_resps = np.exp(log_resps)
        log_resps /= np.sum(log_resps, axis=1, keepdims=True)
        return log_resps

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        effective_n = np.sum(responsibilities, axis=0)
        self.weights = effective_n / n_samples

        for k in range(self.n_components):
            if effective_n[k] < 1e-6:
                print(f"warning: cluster {k} is empty. reinitializing.")
                self.means[k] = X[np.random.choice(len(X))]
                self.covars[k] = np.eye(n_features)
                continue

            self.means[k] = np.dot(responsibilities[:, k], X) / effective_n[k]

            diff = X - self.means[k]
            resp_diag = np.diag(responsibilities[:, k])
            weighted_diff = diff.T @ resp_diag @ diff
            self.covars[k] = weighted_diff / effective_n[k] + np.eye(n_features) * 1e-4

            eigenvalues, eigenvectors = np.linalg.eigh(self.covars[k])
            top_eigenvalues = eigenvalues[-self.n_latent_dims:]
            top_eigenvectors = eigenvectors[:, -self.n_latent_dims:]
            self.latent_covars[k] = np.diag(top_eigenvalues)
            self.covars[k] = (
                top_eigenvectors @ np.diag(top_eigenvalues) @ top_eigenvectors.T
                + np.eye(n_features) * (
                    np.sum(eigenvalues[:-self.n_latent_dims]) / (n_features - self.n_latent_dims)
                )
            )

    def fit(self, X):
        self._initialize_parameters(X)
        log_likelihood = -np.inf
        self.log_likelihoods = []

        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            new_log_likelihood = 0
            for k in range(self.n_components):
                diff = X - self.means[k]
                covar_k = self.covars[k]
                log_det_covar_k = np.log(np.linalg.det(covar_k) + 1e-6)
                inv_covar_k = np.linalg.inv(covar_k + np.eye(X.shape[1]) * 1e-6)
                new_log_likelihood += np.sum(
                    responsibilities[:, k]
                    * (-0.5 * (np.sum(diff @ inv_covar_k * diff, axis=1) + log_det_covar_k))
                )

            self.log_likelihoods.append(new_log_likelihood / X.shape[0])

            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                print(f"converged at iteration {iteration}.")
                break
            log_likelihood = new_log_likelihood

    def predict(self, X):
        return self._e_step(X)


if __name__ == "__main__":
    # Parameters
    n_samples = 700
    n_features = 3
    n_components = 5
    n_latent_dims = 2

    # Generate synthetic 3D data
    def generate_random_covariance(n_features):
        A = np.random.rand(n_features, n_features)
        return np.dot(A, A.T)

    cluster_means = [np.random.rand(n_features) * 10 for _ in range(n_components)]
    cluster_covars = [generate_random_covariance(n_features) for _ in range(n_components)]
    X = np.vstack([
        np.random.multivariate_normal(cluster_means[k], cluster_covars[k], size=n_samples // n_components)
        for k in range(n_components)
    ])

    # Fit MPPCA
    model = MixtureOfPPCA(n_components=n_components, n_latent_dims=n_latent_dims)
    model.fit(X)
    responsibilities = model.predict(X)
    labels = np.argmax(responsibilities, axis=1)

    # Fit classical PCA
    pca = PCA(n_components=n_latent_dims)
    pca.fit(X)

    # Reconstruct data for MPPCA
    def reconstruct_mppca(X, model):
        resp = model.predict(X)
        comps = np.argmax(resp, axis=1)
        Xrec = np.zeros_like(X)
        for i, c in enumerate(comps):
            mu_c = model.means[c]
            cov_c = model.covars[c]
            diff = X[i] - mu_c
            eigenvals, eigenvects = eigh(cov_c)
            idx_desc = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx_desc]
            eigenvects = eigenvects[:, idx_desc]
            lam_top = eigenvals[:model.n_latent_dims]
            U_top = eigenvects[:, :model.n_latent_dims]
            s2 = np.mean(eigenvals[model.n_latent_dims:]) if model.n_latent_dims < len(eigenvals) else 1e-12
            Lamb = U_top.T @ U_top + s2 * np.eye(model.n_latent_dims)
            try:
                M = np.linalg.inv(Lamb) @ U_top.T
            except np.linalg.LinAlgError:
                M = np.linalg.pinv(Lamb) @ U_top.T
            z = M @ diff
            Xrec[i] = mu_c + U_top @ z
        return Xrec

    X_recon_mppca = reconstruct_mppca(X, model)
    mse_mppca = np.mean((X - X_recon_mppca)**2)

    # Reconstruct data for PCA
    X_proj_pca = pca.transform(X)
    X_recon_pca = pca.inverse_transform(X_proj_pca)
    mse_pca = np.mean((X - X_recon_pca)**2)

    print("\n===== Reconstruction MSE =====")
    print(f"PCA (global subspace): {mse_pca:.4f}")
    print(f"MPPCA (local subspaces): {mse_mppca:.4f}")

    # Visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.1, label="Data")

    for k in range(n_components):
        mean = model.means[k]
        eigvals, eigvecs = eigh(model.covars[k])
        top_eigvecs = eigvecs[:, -n_latent_dims:]
        for j in range(n_latent_dims):
            ax.quiver(*mean, *top_eigvecs[:, j], color=f'C{k}', alpha=0.6, linewidth=1)

    global_mean = np.mean(X, axis=0)
    global_pc = pca.components_
    for j in range(n_latent_dims):
        ax.quiver(*global_mean, *global_pc[j], color='k', alpha=0.8, linewidth=2, label=f"Global PC{j+1}")

    ax.set_title("MPPCA vs PCA Subspaces")
    plt.legend()
    plt.show()
