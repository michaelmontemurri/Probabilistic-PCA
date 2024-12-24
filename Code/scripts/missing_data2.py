import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, inv
from sklearn.decomposition import PCA
import random

##############################################################################
# 1. PPCA (Eigen) Implementation
##############################################################################

class PPCAEigen:
    """
    Single-component PPCA using an eigen-decomposition approach.
    We adapt it for handling missing data in a simple manner:
      - We skip missing entries when computing covariances,
      - We fill them in at reconstruction time using the subspace projection.
    """

    def __init__(self, q=2, max_iter=50, tol=1e-5, random_state=42):
        """
        Args:
            q: latent dimension
            max_iter: maximum number of EM-like iterations
            tol: tolerance for convergence checking
            random_state: seed for reproducibility
        """
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

    def _init_params(self, X):
        N, d = X.shape
        self.N, self.d = N, d
        idx = self.rng.choice(N)
        # initialize mean from a random row
        self.mu_ = np.array(X[idx], copy=True)
        # initialize W
        self.W_ = 0.01 * self.rng.randn(d, self.q)
        self.sigma2_ = 1e-2

    def _observed_sample_cov(self, X_centered):
        """
        Compute sample covariance ignoring np.nan entries.
        X_centered shape (N,d). 
        We'll do a double loop to accumulate sums only for observed coords.
        """
        d = X_centered.shape[1]
        S = np.zeros((d,d))
        counts = np.zeros((d,d))
        for n in range(self.N):
            row = X_centered[n]
            obs_idx = np.where(~np.isnan(row))[0]
            for i in obs_idx:
                for j in obs_idx:
                    S[i,j] += row[i]*row[j]
                    counts[i,j] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            S = np.divide(S, counts, out=np.zeros_like(S), where=(counts>0))
        return 0.5*(S+S.T)  # enforce symmetry

    def fit(self, X):
        """
        Fit PPCA ignoring missing data in the covariance step.
        """
        self.X = np.array(X, copy=True)
        self._init_params(self.X)

        ll_old = -np.inf

        for it in range(self.max_iter):
            # 1) compute column means ignoring nans
            col_means = np.zeros(self.d)
            for j in range(self.d):
                col_data = self.X[:,j]
                obs = col_data[~np.isnan(col_data)]
                if len(obs)>0:
                    col_means[j] = np.mean(obs)
                else:
                    col_means[j] = 0.0
            self.mu_ = col_means

            # 2) center ignoring nans
            X_centered = np.array(self.X, copy=True)
            for n in range(self.N):
                for j in range(self.d):
                    if not np.isnan(X_centered[n,j]):
                        X_centered[n,j] -= self.mu_[j]
                    else:
                        X_centered[n,j] = np.nan

            # 3) observed sample covariance
            S = self._observed_sample_cov(X_centered)

            # measure a simple objective: sum of squares of observed data
            ll = 0.0
            for n in range(self.N):
                row = X_centered[n]
                obs = row[~np.isnan(row)]
                ll += np.sum(obs**2)

            if np.abs(ll - ll_old) < self.tol:
                break
            ll_old = ll

            # 4) eigen-decomposition
            vals, vecs = eigh(S)
            idx_desc = np.argsort(vals)[::-1]
            vals = vals[idx_desc]
            vecs = vecs[:, idx_desc]

            lam_top = vals[:self.q]
            U_top = vecs[:, :self.q]

            if self.q < self.d:
                lam_rest = vals[self.q:]
                lam_rest_pos = lam_rest[lam_rest>0]
                self.sigma2_ = np.mean(lam_rest_pos) if len(lam_rest_pos)>0 else 1e-12
            else:
                self.sigma2_ = 1e-12

            loadvals = lam_top - self.sigma2_
            loadvals[loadvals<1e-12] = 1e-12
            self.W_ = U_top @ np.diag(np.sqrt(loadvals))

        return self

    def transform(self, X_in):
        # fill missing with mu
        Xcopy = np.array(X_in, copy=True)
        for i in range(Xcopy.shape[0]):
            for j in range(self.d):
                if np.isnan(Xcopy[i,j]):
                    Xcopy[i,j] = self.mu_[j]

        X_cent = Xcopy - self.mu_
        WtW = self.W_.T @ self.W_
        Lamb = WtW + self.sigma2_*np.eye(self.q)
        invLamb = inv(Lamb)
        M = invLamb @ self.W_.T
        return X_cent @ M.T

    def inverse_transform(self, Z):
        return self.mu_ + Z @ self.W_.T

    def reconstruct(self, X_in):
        Z = self.transform(X_in)
        return self.inverse_transform(Z)


##############################################################################
# 2. PPCA (EM) Implementation
##############################################################################

class PPCAEM:
    """
    Single-component PPCA using an EM approach that handles missing data
    more directly in the E-step and M-step computations.
    """

    def __init__(self, q=2, max_iter=50, tol=1e-5, random_state=42):
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
        self.X = np.array(X, copy=True)
        self._init_params(self.X)

        def partial_log_likelihood(Xrow, mu, W, sigma2):
            # Only for observed coords in Xrow
            obs_idx = np.where(~np.isnan(Xrow))[0]
            if len(obs_idx)==0:
                return 0.0
            row_obs = Xrow[obs_idx]
            d_obs = len(obs_idx)
            W_obs = W[obs_idx]  # shape (d_obs, q)
            C = W_obs @ W_obs.T + sigma2*np.eye(d_obs) + 1e-12*np.eye(d_obs)
            diff = row_obs - mu[obs_idx]
            try:
                logdet = np.linalg.slogdet(C)[1]
                invC = np.linalg.inv(C)
                val = -0.5*(diff @ invC @ diff + logdet + d_obs*np.log(2*np.pi))
                return val
            except:
                return -1e12

        ll_old = -np.inf

        for it in range(self.max_iter):
            # E-step:
            # We'll gather posterior latents ignoring missing coords
            WtW = self.W_.T @ self.W_
            Lamb = WtW + self.sigma2_*np.eye(self.q)
            invLamb = inv(Lamb)
            M = invLamb @ self.W_.T

            # re-estimate mean ignoring nans, column by column
            col_means = np.zeros(self.d)
            for j in range(self.d):
                col_data = self.X[:,j]
                obs = col_data[~np.isnan(col_data)]
                if len(obs)>0:
                    col_means[j] = np.mean(obs)
                else:
                    col_means[j] = 0.0
            self.mu_ = col_means

            S1 = np.zeros((self.d, self.q))
            sum_zzT = np.zeros((self.q, self.q))
            sum_error = 0.0
            count_obs = 0

            for n in range(self.N):
                row = self.X[n]
                obs_idx = np.where(~np.isnan(row))[0]
                if len(obs_idx)==0:
                    continue
                diff = np.array(row, copy=True)
                for j in range(self.d):
                    if np.isnan(diff[j]):
                        diff[j] = self.mu_[j]
                diff -= self.mu_

                Ez_n = M @ diff
                covz = self.sigma2_*invLamb
                Ezz_n = covz + np.outer(Ez_n, Ez_n)

                # accumulate S1
                S1 += np.outer(diff, Ez_n)
                sum_zzT += Ezz_n

                # measure partial error for sigma^2
                # only on observed coords
                recon = self.mu_ + self.W_ @ Ez_n
                local_err = 0
                for idxv in obs_idx:
                    local_err += (row[idxv]-recon[idxv])**2
                sum_error += local_err
                count_obs += len(obs_idx)

            # approximate log-likelihood
            ll = 0.0
            for n in range(self.N):
                ll += partial_log_likelihood(self.X[n], self.mu_, self.W_, self.sigma2_)

            if np.abs(ll - ll_old)< self.tol:
                break
            ll_old = ll

            # M-step
            S1 /= self.N
            sum_zzT /= self.N
            EzzT_inv = inv(sum_zzT + 1e-12*np.eye(self.q))
            self.W_ = S1 @ EzzT_inv
            if count_obs>0:
                self.sigma2_ = sum_error/(count_obs)
            else:
                self.sigma2_ = 1e-12

        return self

    def transform(self, X_in):
        # fill missing
        Xcopy = np.array(X_in, copy=True)
        for i in range(Xcopy.shape[0]):
            for j in range(self.d):
                if np.isnan(Xcopy[i,j]):
                    Xcopy[i,j] = self.mu_[j]
        Xcent = Xcopy - self.mu_

        WtW = self.W_.T @ self.W_
        Lamb = WtW + self.sigma2_*np.eye(self.q)
        invLamb = inv(Lamb)
        M = invLamb @ self.W_.T
        return Xcent @ M.T

    def inverse_transform(self, Z):
        return self.mu_ + Z @ self.W_.T

    def reconstruct(self, X_in):
        Z = self.transform(X_in)
        return self.inverse_transform(Z)


##############################################################################
# 3. Data Generation & Missingness (3D)
##############################################################################

def generate_3d_data(N=200, random_state=123):
    """
    Create a 3D dataset that primarily lies in a 2D subspace + noise.
    For instance: 
      x,y ~ some distribution, z = 2x + 3y + noise
    """
    rng = np.random.RandomState(random_state)
    x = rng.randn(N)*5
    y = rng.randn(N)*3
    z = 2*x + 3*y + rng.randn(N)*2
    return np.vstack([x,y,z]).T  # shape (N, 3)

def remove_data_randomly(X, missing_ratio=0.2, random_state=42):
    """
    Remove 'missing_ratio' fraction of entries from X by setting them to np.nan.
    """
    rng = np.random.RandomState(random_state)
    Xmiss = X.copy()
    N, d = X.shape
    total_entries = N*d
    n_missing = int(missing_ratio*total_entries)
    missing_indices = rng.choice(total_entries, size=n_missing, replace=False)
    for idx in missing_indices:
        row = idx // d
        col = idx % d
        Xmiss[row,col] = np.nan
    return Xmiss

def pca_reconstruction(Xin, d_components=2):
    """
    Classical PCA approach:
      1. fill missing with column means
      2. fit PCA
      3. reconstruct
    """
    Xcopy = np.array(Xin, copy=True)
    N,d = Xcopy.shape
    col_means = np.nanmean(Xcopy, axis=0)
    for i in range(N):
        for j in range(d):
            if np.isnan(Xcopy[i,j]):
                Xcopy[i,j] = col_means[j]

    pca = PCA(n_components=d_components)
    pca.fit(Xcopy)
    Xproj = pca.transform(Xcopy)
    Xrec = pca.inverse_transform(Xproj)
    return Xrec

def measure_missing_mse(Xtrue, Xmiss, Xrecon):
    """
    MSE on positions that are missing in Xmiss but present in Xtrue.
    """
    mask = np.isnan(Xmiss)
    diff = Xrecon[mask] - Xtrue[mask]
    return np.mean(diff**2)

##############################################################################
# 4. Main Demo
##############################################################################

def main():
    # 1) Generate 3D data
    data_3d = generate_3d_data(N=200, random_state=123)

    for ratio in [0.1, 0.25, 0.5, 0.75]:
        missing_ratio = ratio
        data_miss = remove_data_randomly(data_3d, missing_ratio=missing_ratio, random_state=999)

        # 2) Reconstruct with classical PCA
        pca_rec = pca_reconstruction(data_miss, d_components=2)
        mse_pca = measure_missing_mse(data_3d, data_miss, pca_rec)

        # 3) Reconstruct with PPCA (Eigen)
        ppca_eig = PPCAEigen(q=2, max_iter=50, tol=1e-4, random_state=123)
        ppca_eig.fit(data_miss)
        ppca_eig_rec = ppca_eig.reconstruct(data_miss)
        mse_ppca_eig = measure_missing_mse(data_3d, data_miss, ppca_eig_rec)

        # 4) Reconstruct with PPCA (EM)
        ppca_em = PPCAEM(q=2, max_iter=50, tol=1e-4, random_state=123)
        ppca_em.fit(data_miss)
        ppca_em_rec = ppca_em.reconstruct(data_miss)
        mse_ppca_em = measure_missing_mse(data_3d, data_miss, ppca_em_rec)

        print(f"Missing ratio: {missing_ratio*100:.1f}%")
        print("Reconstruction MSE on missing entries:")
        print(f"  PCA:            {mse_pca:.4f}")
        print(f"  PPCA (Eigen):   {mse_ppca_eig:.4f}")
        print(f"  PPCA (EM):      {mse_ppca_em:.4f}")

        # Visual: We'll do 2x2 subplots in 3D
        # (0,0) => Original with missing indicated
        # (0,1) => PCA
        # (1,0) => PPCA (Eigen)
        # (1,1) => PPCA (EM)

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10,8))

        # original data + missing
        ax0 = fig.add_subplot(221, projection='3d')
        ax0.set_title("Original Data + Missing")
        # show all points in grey
        ax0.scatter(data_3d[:,0], data_3d[:,1], data_3d[:,2], c='grey', alpha=0.4, label='All data')
        # highlight points that had missing entries
        mask_missing = np.isnan(data_miss)
        missing_idx = np.unique(np.where(mask_missing)[0])
        ax0.scatter(data_3d[missing_idx,0], data_3d[missing_idx,1], data_3d[missing_idx,2],
                    facecolors='none', edgecolors='red', s=50, label='Missing coords')
        ax0.legend()

        # PCA
        ax1 = fig.add_subplot(222, projection='3d')
        ax1.set_title("PCA Reconstruction")
        ax1.scatter(pca_rec[:,0], pca_rec[:,1], pca_rec[:,2], c='blue', alpha=0.6)
        ax1.scatter(pca_rec[missing_idx,0], pca_rec[missing_idx,1], pca_rec[missing_idx,2],
                    facecolors='none', edgecolors='red', s=50)

        # PPCA (Eigen)
        ax2 = fig.add_subplot(223, projection='3d')
        ax2.set_title("PPCA (Eigen) Reconstruction")
        ax2.scatter(ppca_eig_rec[:,0], ppca_eig_rec[:,1], ppca_eig_rec[:,2],
                    c='green', alpha=0.6)
        ax2.scatter(ppca_eig_rec[missing_idx,0], ppca_eig_rec[missing_idx,1], ppca_eig_rec[missing_idx,2],
                    facecolors='none', edgecolors='red', s=50)

        # PPCA (EM)
        ax3 = fig.add_subplot(224, projection='3d')
        ax3.set_title("PPCA (EM) Reconstruction")
        ax3.scatter(ppca_em_rec[:,0], ppca_em_rec[:,1], ppca_em_rec[:,2],
                    c='purple', alpha=0.6)
        ax3.scatter(ppca_em_rec[missing_idx,0], ppca_em_rec[missing_idx,1], ppca_em_rec[missing_idx,2],
                    facecolors='none', edgecolors='red', s=50)

        for ax in [ax0,ax1,ax2,ax3]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        plt.suptitle("3D Missing Data: PCA vs PPCA (Eigen) vs PPCA (EM)")
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    main()
