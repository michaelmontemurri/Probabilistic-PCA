import numpy as np
from numpy.linalg import eigh, inv, slogdet

###############################################################################
#                         1) PPCA via Eigen-decomposition                     #
###############################################################################

class PPCAEigen:
    """
    Single-component PPCA using an eigen-decomposition approach.
    
    This implementation:
      - Approximates missing data by skipping missing entries in the covariance estimate.
      - Uses a simple "sum of squares" convergence check, which is not the exact LL.
      - Reconstructs missing entries by projecting onto the learned subspace.
      
    References:
    Tipping & Bishop, 1999 (Probabilistic PCA). 
    """

    def __init__(self, q=2, max_iter=50, tol=1e-5, random_state=42):
        """
        Args:
            q: latent dimension
            max_iter: maximum number of EM-like iterations
            tol: convergence tolerance
            random_state: seed for reproducibility
        """
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

    def _init_params(self, X):
        """
        Initialize the mean (mu_), loading matrix (W_), and noise variance (sigma2_).
        """
        N, d = X.shape
        self.N, self.d = N, d
        # pick a random row to initialize mean
        idx = self.rng.choice(N)
        self.mu_ = np.array(X[idx], copy=True)
        # small random W
        self.W_ = 0.01 * self.rng.randn(d, self.q)
        self.sigma2_ = 1e-2

    def _observed_sample_cov(self, X_centered):
        """
        Compute sample covariance ignoring missing entries (np.nan).
        
        Naive O(N d^2) approach: 
        For each row, only accumulate contributions for observed pairs.
        """
        d = X_centered.shape[1]
        S = np.zeros((d, d))
        counts = np.zeros((d, d))

        for n in range(self.N):
            row = X_centered[n]
            obs_idx = np.where(~np.isnan(row))[0]
            for i in obs_idx:
                for j in obs_idx:
                    S[i, j] += row[i]*row[j]
                    counts[i, j] += 1
        
        # safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            S = np.divide(S, counts, out=np.zeros_like(S), where=(counts > 0))
        # enforce symmetry
        S = 0.5 * (S + S.T)
        return S

    def fit(self, X):
        """
        Fit PPCA model.  
        
        Steps per iteration:
          1) Estimate mean from observed values in each column.
          2) Center data (subtract mean on observed entries).
          3) Compute sample covariance ignoring missing.
          4) Eigen-decompose, update W_, sigma2_.
          5) Check convergence by comparing a naive "sum of squares" measure.
        """
        self.X = np.array(X, copy=True)
        self._init_params(self.X)

        ll_old = -np.inf
        for it in range(self.max_iter):
            # 1) estimate mean ignoring nans
            col_means = np.zeros(self.d)
            for j in range(self.d):
                col_data = self.X[:, j]
                obs = col_data[~np.isnan(col_data)]
                col_means[j] = np.mean(obs) if len(obs) > 0 else 0.0
            self.mu_ = col_means

            # 2) center data ignoring nans
            X_centered = np.array(self.X, copy=True)
            for n in range(self.N):
                for j in range(self.d):
                    if not np.isnan(X_centered[n, j]):
                        X_centered[n, j] -= self.mu_[j]
                    else:
                        X_centered[n, j] = np.nan

            # 3) sample covariance for observed entries
            S = self._observed_sample_cov(X_centered)

            # 4) measure a naive "objective": sum of squares of observed data
            #    (not the true negative log-likelihood)
            ll = 0.0
            for n in range(self.N):
                row = X_centered[n]
                obs = row[~np.isnan(row)]
                ll += np.sum(obs**2)

            # check convergence
            if np.abs(ll - ll_old) < self.tol:
                break
            ll_old = ll

            # 5) eigen-decomposition
            vals, vecs = eigh(S)
            idx_desc = np.argsort(vals)[::-1]
            vals = vals[idx_desc]
            vecs = vecs[:, idx_desc]

            # top q
            lam_top = vals[:self.q]
            U_top = vecs[:, :self.q]

            # estimate sigma^2 from the average of discarded eigenvalues
            if self.q < self.d:
                lam_rest = vals[self.q:]
                lam_rest_pos = lam_rest[lam_rest > 0]
                self.sigma2_ = np.mean(lam_rest_pos) if len(lam_rest_pos) > 0 else 1e-12
            else:
                # if q == d, no "discarded" eigenvalues, fallback
                self.sigma2_ = 1e-12

            # clamp negative loadings for numerical safety
            loadvals = lam_top - self.sigma2_
            loadvals[loadvals < 1e-12] = 1e-12

            # W = U * sqrt(diag(loadvals))
            self.W_ = U_top @ np.diag(np.sqrt(loadvals))

        return self

    def transform(self, X_in):
        """
        Project data into the latent space z = (W^T W + sigma^2 I)^{-1} W^T (x - mu).
        
        Missing features are imputed with the learned mean mu_.
        """
        Xcopy = np.array(X_in, copy=True)
        for i in range(Xcopy.shape[0]):
            for j in range(self.d):
                if np.isnan(Xcopy[i, j]):
                    Xcopy[i, j] = self.mu_[j]
        # center
        X_cent = Xcopy - self.mu_
        # invert the (W^T W + sigma^2 I) matrix
        WtW = self.W_.T @ self.W_
        Lamb = WtW + self.sigma2_ * np.eye(self.q)
        invLamb = inv(Lamb)
        # M = inv(Lamb) W^T
        M = invLamb @ self.W_.T
        return X_cent @ M.T

    def inverse_transform(self, Z):
        """
        Map latent codes Z back to data space via X = mu + Z W^T.
        """
        return self.mu_ + Z @ self.W_.T

    def reconstruct(self, X_in):
        """
        Reconstruct X_in (filling missing entries) using the PPCA model.
        """
        Z = self.transform(X_in)
        return self.inverse_transform(Z)


###############################################################################
#                          2) PPCA via EM Algorithm                           #
###############################################################################

class PPCAEM:
    """
    Single-component PPCA using an EM approach.
    This version:
      - In the E-step, missing data are simply imputed with mu_ for reconstruction,
        so it's not the exact blockwise formula for missing data.
      - The partial log-likelihood is computed only over observed indices, used as
        a convergence check.
    """

    def __init__(self, q=2, max_iter=50, tol=1e-5, random_state=42):
        """
        Args:
            q: latent dimension
            max_iter: maximum number of EM iterations
            tol: convergence tolerance on log-likelihood
            random_state: for reproducibility
        """
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

    def _init_params(self, X):
        N, d = X.shape
        self.N, self.d = N, d
        idx = self.rng.choice(N)
        self.mu_ = np.array(X[idx], copy=True)
        self.W_ = 0.01 * self.rng.randn(d, self.q)
        self.sigma2_ = 1e-2

    def fit(self, X):
        """
        Fit the PPCA model with EM:
          - E-step: compute posterior latents for each sample (imputing missing features).
          - M-step: update mu, W, sigma2 from the posterior statistics.
          - Evaluate partial log-likelihood on observed indices.
        """
        self.X = np.array(X, copy=True)
        self._init_params(self.X)

        def partial_log_likelihood(Xrow, mu, W, sigma2):
            """
            Compute log p(x_obs) = -0.5 [ (x_obs - mu_obs)^T C_inv (x_obs - mu_obs)
                                          + log det(C) + d_obs log(2 pi) ]
            where C = W_obs W_obs^T + sigma2 I.
            """
            obs_idx = np.where(~np.isnan(Xrow))[0]
            if len(obs_idx) == 0:
                # if entire row is missing, skip
                return 0.0
            row_obs = Xrow[obs_idx]
            d_obs = len(obs_idx)
            W_obs = W[obs_idx]  # shape (d_obs, q)
            # C = W_obs W_obs^T + sigma2 I
            C = W_obs @ W_obs.T + sigma2 * np.eye(d_obs)
            diff = row_obs - mu[obs_idx]
            # numerically stable slogdet
            signC, logdetC = slogdet(C)
            if signC <= 0:
                return -1e12  # extremely negative if degenerate
            # Mahalanobis term
            try:
                invC = inv(C)
            except:
                return -1e12
            val = -0.5 * (diff @ invC @ diff + logdetC + d_obs * np.log(2.0 * np.pi))
            return val

        ll_old = -np.inf
        for it in range(self.max_iter):
            # E-step
            # 1) update mu from observed columns
            col_means = np.zeros(self.d)
            for j in range(self.d):
                col_data = self.X[:, j]
                obs = col_data[~np.isnan(col_data)]
                col_means[j] = np.mean(obs) if len(obs) > 0 else 0.0
            self.mu_ = col_means

            # 2) precompute terms for posterior latents
            WtW = self.W_.T @ self.W_
            Lamb = WtW + self.sigma2_ * np.eye(self.q)
            invLamb = inv(Lamb)
            M = invLamb @ self.W_.T

            # accumulators for M-step
            S1 = np.zeros((self.d, self.q))    # sum of x_n z_n^T
            sum_zzT = np.zeros((self.q, self.q))
            sum_error = 0.0
            count_obs = 0

            # for computing partial log-likelihood
            ll = 0.0

            for n in range(self.N):
                row = self.X[n]
                obs_idx = np.where(~np.isnan(row))[0]

                # fill missing values with mu
                diff = np.array(row, copy=True)
                for j in range(self.d):
                    if np.isnan(diff[j]):
                        diff[j] = self.mu_[j]
                # center
                diff -= self.mu_

                # posterior latent mean z_n = M @ (x_n - mu)
                Ez_n = M @ diff
                # posterior latent cov covz = sigma2 * invLamb
                covz = self.sigma2_ * invLamb
                Ezz_n = covz + np.outer(Ez_n, Ez_n)

                # accumulate for W update
                S1 += np.outer(diff, Ez_n)
                sum_zzT += Ezz_n

                # reconstruction error on observed coordinates
                recon = self.mu_ + self.W_ @ Ez_n
                if len(obs_idx) > 0:
                    local_err = np.sum((row[obs_idx] - recon[obs_idx])**2)
                    sum_error += local_err
                    count_obs += len(obs_idx)

                # partial likelihood
                ll += partial_log_likelihood(self.X[n], self.mu_, self.W_, self.sigma2_)

            # check convergence
            if np.abs(ll - ll_old) < self.tol:
                break
            ll_old = ll

            # M-step
            # update W
            S1 /= self.N
            sum_zzT /= self.N
            # invert (E[zz^T]) for update
            EzzT_inv = inv(sum_zzT + 1e-12 * np.eye(self.q))
            self.W_ = S1 @ EzzT_inv

            # update sigma^2
            if count_obs > 0:
                self.sigma2_ = sum_error / count_obs
            else:
                self.sigma2_ = 1e-12

        return self

    def transform(self, X_in):
        """
        Project data into latent space. Missing is imputed with mu_.
        """
        Xcopy = np.array(X_in, copy=True)
        for i in range(Xcopy.shape[0]):
            for j in range(self.d):
                if np.isnan(Xcopy[i, j]):
                    Xcopy[i, j] = self.mu_[j]
        Xcent = Xcopy - self.mu_

        WtW = self.W_.T @ self.W_
        Lamb = WtW + self.sigma2_ * np.eye(self.q)
        invLamb = inv(Lamb)
        M = invLamb @ self.W_.T
        return Xcent @ M.T

    def inverse_transform(self, Z):
        """
        Map latent codes back to data space: x = mu + z W^T.
        """
        return self.mu_ + Z @ self.W_.T

    def reconstruct(self, X_in):
        """
        Reconstruct X_in using the learned PPCA model.
        """
        Z = self.transform(X_in)
        return self.inverse_transform(Z)


###############################################################################
#                   3) Mixture of PPCA (closer to Tipping & Bishop)           #
###############################################################################

class MixtureOfPPCA:
    """
    Mixture of PPCA model (Tipping & Bishop, 1999).
    
    We maintain for each component k:
      - Weight pi_k
      - Mean mu_k (d-dimensional)
      - Loading matrix W_k (d x q)
      - Noise variance sigma2_k (scalar)
    
    The covariance for component k is: Sigma_k = W_k W_k^T + sigma2_k I.
    
    In the E-step:
      - Compute responsibilities gamma_{nk} = p(k|x_n).
      - Also compute the posterior distribution of latent variables Z_{nk} for each mixture component.
    In the M-step:
      - Update pi_k, mu_k, W_k, sigma2_k using formulae analogous to the single-component PPCA,
        weighted by gamma_{nk}.
    """

    def __init__(self, n_components=2, q=2, max_iter=100, tol=1e-4, random_state=42):
        """
        Args:
            n_components: number of mixture components
            q: latent dimension for each component
            max_iter: maximum EM iterations
            tol: convergence tolerance on log-likelihood
            random_state: for reproducibility
        """
        self.n_components = n_components
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

    def _init_params(self, X):
        """
        Initialize mixture weights, means, W, and sigma2 for each component.
        """
        N, d = X.shape
        self.N, self.d = N, d

        # mixture weights
        self.weights_ = np.ones(self.n_components) / self.n_components

        # choose random distinct rows for means (or fallback to random)
        rand_idx = self.rng.choice(N, self.n_components, replace=False)
        self.means_ = X[rand_idx].copy()

        # W matrices: each is d x q
        self.W_ = 0.01 * self.rng.randn(self.n_components, d, self.q)

        # noise variances
        self.sigma2_ = np.full(self.n_components, 1e-2)

    def _estimate_log_gaussian(self, X, k):
        """
        Compute log of N(x | mu_k, W_k W_k^T + sigma2_k I) for each row in X.
        For numerical stability, we do:
          log p(x_n | k) = -0.5 [ (x_n - mu_k)^T Sigma_k^{-1} (x_n - mu_k)
                                  + log det(Sigma_k) + d log(2 pi) ]
        """
        diff = X - self.means_[k]
        Wk = self.W_[k]       # shape d x q
        sig2 = self.sigma2_[k]
        # Sigma_k = W_k W_k^T + sigma2_k I
        # We'll use the Woodbury identity to invert Sigma_k more stably:
        #   inv(Sigma_k) = (1 / sigma2_k) I - (1 / sigma2_k) W_k (I + 1/sigma2_k W_k^T W_k)^{-1} W_k^T (1 / sigma2_k)
        # but for simplicity we can do a direct approach for now, if d not too large.
        # direct approach:
        Sigma_k = Wk @ Wk.T + sig2 * np.eye(self.d)

        # regularize for numerical safety
        Sigma_k += 1e-12 * np.eye(self.d)

        sign_det, logdet = slogdet(Sigma_k)
        if sign_det <= 0:
            # degenerate covariance
            return np.full(X.shape[0], -1e12)

        invSigma_k = inv(Sigma_k)

        # Mahalanobis distance
        mdist = np.sum(diff @ invSigma_k * diff, axis=1)  # row-wise x^T invSigma x
        # log p(x_n | k)
        logp = -0.5 * (mdist + logdet + self.d * np.log(2.0 * np.pi))
        return logp

    def _e_step(self, X):
        """
        E-step:
          1) For each k, compute log p(x_n | k).
          2) Combine with log weights to get log responsibilities.
          3) Exponentiate, normalize.
          4) Also compute posterior latents Z_{nk}.
        """
        N, d = X.shape

        # compute log p(x | k) for each component k
        log_pdf = np.zeros((N, self.n_components))
        for k in range(self.n_components):
            log_pdf[:, k] = self._estimate_log_gaussian(X, k)

        # add log weights
        log_pdf += np.log(self.weights_ + 1e-12)

        # subtract max for numerical stability
        max_logpdf = np.max(log_pdf, axis=1, keepdims=True)
        log_pdf -= max_logpdf
        resp_unnorm = np.exp(log_pdf)
        # normalize
        resp = resp_unnorm / np.sum(resp_unnorm, axis=1, keepdims=True)

        # Now, compute posterior latents (means and cov) for each (n,k)
        # z_{nk} = M_k (x_n - mu_k),  M_k = (W_k^T W_k + sigma2_k I)^{-1} W_k^T
        Ez = np.zeros((N, self.n_components, self.q))
        EzzT = np.zeros((N, self.n_components, self.q, self.q))

        for k in range(self.n_components):
            Wk = self.W_[k]  # shape (d, q)
            sig2 = self.sigma2_[k]
            Lamb = Wk.T @ Wk + sig2 * np.eye(self.q)
            invLamb = inv(Lamb)
            M_k = invLamb @ Wk.T  # shape (q, d)

            for n in range(N):
                diff = X[n] - self.means_[k]
                Ez_nk = M_k @ diff
                # posterior cov of z_nk
                covz_nk = sig2 * invLamb
                # store
                Ez[n, k] = Ez_nk
                EzzT[n, k] = covz_nk + np.outer(Ez_nk, Ez_nk)

        return resp, Ez, EzzT

    def _m_step(self, X, resp, Ez, EzzT):
        """
        M-step:
          - Update pi_k = (1/N) sum_n gamma_{nk}
          - Update mu_k, W_k, sigma2_k similarly to single PPCA, weighted by gamma_{nk}.
        """
        N, d = X.shape
        gamma_k = np.sum(resp, axis=0)  # shape (K,)
        self.weights_ = gamma_k / N

        # for each component k
        for k in range(self.n_components):
            # if cluster has negligible weight, reinit
            if gamma_k[k] < 1e-12:
                # re-initialize
                self.means_[k] = X[self.rng.choice(N)]
                self.W_[k] = 0.01 * self.rng.randn(d, self.q)
                self.sigma2_[k] = 1e-2
                self.weights_[k] = 1.0 / self.n_components
                continue

            # Weighted means
            # mu_k = (1 / gamma_k) sum_n gamma_{nk} x_n
            sum_resp_xn = np.sum(resp[:, k, np.newaxis] * X, axis=0)
            self.means_[k] = sum_resp_xn / gamma_k[k]

            # Now update W_k, sigma2_k
            # S1_k = (1 / gamma_k) sum_n gamma_{nk} (x_n - mu_k) z_{nk}^T
            # Ezz_k = (1 / gamma_k) sum_n gamma_{nk} E[z_{nk} z_{nk}^T]
            S1_k = np.zeros((d, self.q))
            Ezz_k = np.zeros((self.q, self.q))
            sum_err = 0.0

            for n in range(N):
                diff = X[n] - self.means_[k]
                S1_k += resp[n, k] * np.outer(diff, Ez[n, k])
                Ezz_k += resp[n, k] * EzzT[n, k]

            S1_k /= gamma_k[k]
            Ezz_k /= gamma_k[k]

            # W_k = S1_k Ezz_k^{-1}
            Ezz_inv = inv(Ezz_k + 1e-12 * np.eye(self.q))
            W_k_new = S1_k @ Ezz_inv
            self.W_[k] = W_k_new

            # compute sigma2_k
            # sigma2_k = (1 / (gamma_k * d)) sum_n gamma_{nk} sum_over_d [ (x_nk - mu_k) - W_k z_nk ]^2
            # We'll do a more direct approach:
            sum_sq = 0.0
            count_tot = gamma_k[k] * d  # "effective" count in that cluster

            for n in range(N):
                diff = X[n] - self.means_[k]
                recon = self.means_[k] + W_k_new @ Ez[n, k]
                sq_err = np.sum((diff - (recon - self.means_[k]))**2)
                sum_sq += resp[n, k] * sq_err

            # note: we can also do a trace approach with EzzT, but let's keep it straightforward
            self.sigma2_[k] = sum_sq / (count_tot + 1e-12)

    def fit(self, X):
        """
        Full EM loop for mixture of PPCA.
        """
        X = np.asarray(X)
        self._init_params(X)

        prev_ll = -np.inf
        self.log_likelihoods_ = []

        for i in range(self.max_iter):
            # E-step
            resp, Ez, EzzT = self._e_step(X)
            # M-step
            self._m_step(X, resp, Ez, EzzT)

            # compute new log-likelihood
            log_pdf = np.zeros((self.N, self.n_components))
            for k in range(self.n_components):
                log_pdf[:, k] = self._estimate_log_gaussian(X, k) + np.log(self.weights_[k] + 1e-12)
            # log-sum-exp
            max_logpdf = np.max(log_pdf, axis=1, keepdims=True)
            ll_row = max_logpdf + np.log(np.sum(np.exp(log_pdf - max_logpdf), axis=1, keepdims=True))
            ll_total = np.sum(ll_row)

            self.log_likelihoods_.append(ll_total)

            if np.abs(ll_total - prev_ll) < self.tol:
                print(f"Converged at iteration {i}.")
                break
            prev_ll = ll_total

        return self

    def predict(self, X):
        """
        Compute posterior responsibilities gamma_{nk} for new data.
        """
        resp, Ez, EzzT = self._e_step(X)
        return resp

    def transform(self, X):
        """
        Return the posterior mean of the latent variable for each mixture component, 
        weighted by the responsibilities. If you just want the component with highest 
        posterior responsibility, you can adapt accordingly.
        """
        resp, Ez, _ = self._e_step(X)
        # Weighted average over components
        # shape of Ez: (N, K, q)
        # shape of resp: (N, K)
        # Weighted latent = sum_k gamma_{nk} Ez_{nk}
        # but note we do row-wise normalization by sum_k gamma_{nk}=1
        # which is already 1, so:
        Z_post = np.einsum('nk,nkq->nq', resp, Ez)
        return Z_post

    def inverse_transform(self, Z, component=0):
        """
        Map latent codes back to data space using a single chosen component.
        Typically you'd choose the component with highest responsibility, 
        or do a weighted average. This is a simple version that uses one component.
        
        Args:
            Z: shape (N, q)
            component: int, which mixture component to use
        """
        return self.means_[component] + Z @ self.W_[component].T

    def reconstruct(self, X):
        """
        For each point x_n, pick the component k with the highest responsibility, 
        then reconstruct using that component's W_k and mu_k.
        """
        resp = self.predict(X)  # (N, K)
        comp_idx = np.argmax(resp, axis=1)
        X_rec = np.zeros_like(X)
        # E-step to get posterior latent Ez
        _, Ez, _ = self._e_step(X)

        for n in range(X.shape[0]):
            k = comp_idx[n]
            X_rec[n] = self.means_[k] + self.W_[k] @ Ez[n, k]
        return X_rec
