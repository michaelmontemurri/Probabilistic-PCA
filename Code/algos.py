import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# class for mixtures of probabilistic pca
class MixtureOfPPCA:
    def __init__(self, n_components, n_latent_dims, max_iter=100, tol=1e-4):
        # initialize the basic parameters
        self.n_components = n_components  # number of mixture components
        self.n_latent_dims = n_latent_dims  # number of latent dimensions
        self.max_iter = max_iter  # max number of iterations for em
        self.tol = tol  # convergence tolerance
        self.weights = None  # mixture weights
        self.means = None  # component means
        self.covars = None  # covariance matrices
        self.latent_covars = None  # latent variable covariances

    def _initialize_parameters(self, X):
        # randomly initialize weights, means, and covariances
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covars = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.latent_covars = np.array([np.eye(self.n_latent_dims) for _ in range(self.n_components)])

    def _e_step(self, X):
        # compute responsibilities in the e-step
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
        # normalize log responsibilities
        max_log_resps = np.max(log_resps, axis=1, keepdims=True)
        log_resps -= max_log_resps
        log_resps = np.exp(log_resps)
        log_resps /= np.sum(log_resps, axis=1, keepdims=True)
        return log_resps

    def _m_step(self, X, responsibilities):
        # update parameters in the m-step
        n_samples, n_features = X.shape
        effective_n = np.sum(responsibilities, axis=0)
        self.weights = effective_n / n_samples
        for k in range(self.n_components):
            if effective_n[k] < 1e-6:  # reinitialize empty clusters
                print(f"warning: cluster {k} is empty. reinitializing.")
                self.means[k] = X[np.random.choice(len(X))]
                self.covars[k] = np.eye(n_features)
                continue
            # update means
            self.means[k] = np.dot(responsibilities[:, k], X) / effective_n[k]
            # update covariances
            diff = X - self.means[k]
            resp_diag = np.diag(responsibilities[:, k])
            weighted_diff = diff.T @ resp_diag @ diff
            self.covars[k] = weighted_diff / effective_n[k] + np.eye(n_features) * 1e-4
            # update latent covariances via eigen-decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(self.covars[k])
            top_eigenvalues = eigenvalues[-self.n_latent_dims:]
            top_eigenvectors = eigenvectors[:, -self.n_latent_dims:]
            self.latent_covars[k] = np.diag(top_eigenvalues)
            self.covars[k] = (
                top_eigenvectors @ np.diag(top_eigenvalues) @ top_eigenvectors.T
                + np.eye(n_features) * (np.sum(eigenvalues[:-self.n_latent_dims]) / (n_features - self.n_latent_dims))
            )

    def fit(self, X):
        # run the em algorithm to fit the model
        self._initialize_parameters(X)
        log_likelihood = -np.inf
        self.log_likelihoods = []
        for iteration in range(self.max_iter):
            # e-step: calculate responsibilities
            responsibilities = self._e_step(X)
            # m-step: update parameters
            self._m_step(X, responsibilities)
            # compute the log-likelihood
            new_log_likelihood = 0
            for k in range(self.n_components):
                diff = X - self.means[k]
                covar_k = self.covars[k]
                log_det_covar_k = np.log(np.linalg.det(covar_k))
                inv_covar_k = np.linalg.inv(covar_k)
                new_log_likelihood += np.sum(
                    responsibilities[:, k]
                    * (-0.5 * (np.sum(diff @ inv_covar_k * diff, axis=1) + log_det_covar_k))
                )
            self.log_likelihoods.append(new_log_likelihood / X.shape[0])
            # check convergence
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                print(f"converged at iteration {iteration}.")
                break
            log_likelihood = new_log_likelihood

    def predict(self, X):
        return self._e_step(X)
