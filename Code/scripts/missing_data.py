import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, inv
from sklearn.decomposition import PCA
from algos import PPCAEigen, PPCAEM

def generate_3d_data(N=200, random_state=123):
    """
    Generate a 3D dataset that primarily lies in a 2D subspace plus noise.

    Specifically:
        x, y ~ Normal(0, scaled) 
        z = 2*x + 3*y + small random noise
    
    Args:
        N (int): number of data points to generate.
        random_state (int): seed for reproducibility.

    Returns:
        X (np.ndarray): shape (N, 3), 3D data array.
    """
    rng = np.random.RandomState(random_state)
    x = rng.randn(N) * 5      # broaden X dimension
    y = rng.randn(N) * 3      # broaden Y dimension
    z = 2*x + 3*y + rng.randn(N)*2  # Z is linear combination + noise
    return np.vstack([x, y, z]).T   # shape (N, 3)

def remove_data_randomly(X, missing_ratio=0.2, random_state=42):
    """
    Randomly remove a fraction of the entries in X by setting them to np.nan.

    Args:
        X (np.ndarray): original data, shape (N, d).
        missing_ratio (float): fraction of total entries to replace with np.nan.
        random_state (int): seed for reproducibility.

    Returns:
        Xmiss (np.ndarray): copy of X with missing_ratio fraction of entries set to np.nan.
    """
    rng = np.random.RandomState(random_state)
    Xmiss = X.copy()
    N, d = Xmiss.shape
    total_entries = N * d
    n_missing = int(missing_ratio * total_entries)

    # Randomly choose which entries to set to np.nan
    missing_indices = rng.choice(total_entries, size=n_missing, replace=False)
    
    for idx in missing_indices:
        row = idx // d
        col = idx % d
        Xmiss[row, col] = np.nan

    return Xmiss

def pca_reconstruction(Xin, d_components=2):
    """
    Reconstruct data with classical PCA by:
      1) Imputing missing values with the column mean,
      2) Fitting PCA on the completed data,
      3) Inversely transforming the principal components (reconstruction).

    Args:
        Xin (np.ndarray): shape (N, d), data with possible np.nan entries.
        d_components (int): number of principal components to use.

    Returns:
        Xrec (np.ndarray): PCA-based reconstruction, shape (N, d).
    """
    Xcopy = np.array(Xin, copy=True)
    N, d = Xcopy.shape

    # 1) Impute missing with column means
    col_means = np.nanmean(Xcopy, axis=0)
    for i in range(N):
        for j in range(d):
            if np.isnan(Xcopy[i, j]):
                Xcopy[i, j] = col_means[j]

    # 2) Fit PCA
    pca = PCA(n_components=d_components)
    pca.fit(Xcopy)

    # 3) Reconstruct
    Xproj = pca.transform(Xcopy)
    Xrec = pca.inverse_transform(Xproj)
    return Xrec

def measure_missing_mse(Xtrue, Xmiss, Xrecon):
    """
    Compute the Mean Squared Error (MSE) only on entries that are missing in Xmiss 
    but present in Xtrue.

    Args:
        Xtrue (np.ndarray): the true data, shape (N, d).
        Xmiss (np.ndarray): the data with missing entries (np.nan), same shape.
        Xrecon (np.ndarray): the reconstructed data, same shape.

    Returns:
        float: mean squared error over the missing entries.
    """
    # mask of missing entries
    mask = np.isnan(Xmiss)
    # difference only on missing positions
    diff = Xrecon[mask] - Xtrue[mask]
    return np.mean(diff**2)

def missing_data():
    """
    Main function that:
      1) Generates a 3D dataset with slight noise.
      2) Removes various fractions of data at random.
      3) Reconstructs missing values using:
         - Classical PCA
         - PPCA (Eigen)
         - PPCA (EM)
      4) Compares MSE of reconstruction on missing entries.
      5) Visualizes the original data (marking missing) and each reconstruction.
    """
    # 1) Generate the original 3D data
    data_3d = generate_3d_data(N=200, random_state=123)

    # We'll test several missing-data ratios
    for ratio in [0.01, 0.05, 0.1, 0.25]:
        missing_ratio = ratio
        
        # 2) Remove data randomly
        data_miss = remove_data_randomly(data_3d, missing_ratio=missing_ratio, random_state=999)

        # 3a) Reconstruct using classical PCA
        pca_rec = pca_reconstruction(data_miss, d_components=2)
        mse_pca = measure_missing_mse(data_3d, data_miss, pca_rec)

        # 3b) Reconstruct using PPCA (Eigen)
        ppca_eig = PPCAEigen(q=2, max_iter=50, tol=1e-4, random_state=123)
        ppca_eig.fit(data_miss)
        ppca_eig_rec = ppca_eig.reconstruct(data_miss)
        mse_ppca_eig = measure_missing_mse(data_3d, data_miss, ppca_eig_rec)

        # 3c) Reconstruct using PPCA (EM)
        ppca_em = PPCAEM(q=2, max_iter=50, tol=1e-4, random_state=123)
        ppca_em.fit(data_miss)
        ppca_em_rec = ppca_em.reconstruct(data_miss)
        mse_ppca_em = measure_missing_mse(data_3d, data_miss, ppca_em_rec)

        # Print out MSE results
        print(f"Missing ratio: {missing_ratio*100:.1f}%")
        print("Reconstruction MSE on missing entries:")
        print(f"  PCA:            {mse_pca:.4f}")
        print(f"  PPCA (Eigen):   {mse_ppca_eig:.4f}")
        print(f"  PPCA (EM):      {mse_ppca_em:.4f}")
        print("--------------------------------------------------")

        # 4) Visualization in 3D subplots
        fig = plt.figure(figsize=(10, 8))
        from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

        # Subplot (2x2): Original data with missing highlighted
        ax0 = fig.add_subplot(221, projection='3d')
        ax0.set_title("Original + Missing")
        ax0.scatter(data_3d[:,0], data_3d[:,1], data_3d[:,2],
                    c='grey', alpha=0.4, label='All data')
        # highlight missing points in red
        mask_missing = np.isnan(data_miss)
        missing_idx = np.unique(np.where(mask_missing)[0])
        ax0.scatter(data_3d[missing_idx,0], data_3d[missing_idx,1], data_3d[missing_idx,2],
                    facecolors='none', edgecolors='red', s=50, label='Missing coords')
        ax0.legend()

        # Subplot: PCA reconstruction
        ax1 = fig.add_subplot(222, projection='3d')
        ax1.set_title("PCA Reconstruction")
        ax1.scatter(pca_rec[:,0], pca_rec[:,1], pca_rec[:,2],
                    c='blue', alpha=0.6)
        ax1.scatter(pca_rec[missing_idx,0], pca_rec[missing_idx,1], pca_rec[missing_idx,2],
                    facecolors='none', edgecolors='red', s=50)

        # Subplot: PPCA (Eigen) reconstruction
        ax2 = fig.add_subplot(223, projection='3d')
        ax2.set_title("PPCA (Eigen)")
        ax2.scatter(ppca_eig_rec[:,0], ppca_eig_rec[:,1], ppca_eig_rec[:,2],
                    c='green', alpha=0.6)
        ax2.scatter(ppca_eig_rec[missing_idx,0], ppca_eig_rec[missing_idx,1], ppca_eig_rec[missing_idx,2],
                    facecolors='none', edgecolors='red', s=50)

        # Subplot: PPCA (EM) reconstruction
        ax3 = fig.add_subplot(224, projection='3d')
        ax3.set_title("PPCA (EM)")
        ax3.scatter(ppca_em_rec[:,0], ppca_em_rec[:,1], ppca_em_rec[:,2],
                    c='purple', alpha=0.6)
        ax3.scatter(ppca_em_rec[missing_idx,0], ppca_em_rec[missing_idx,1], ppca_em_rec[missing_idx,2],
                    facecolors='none', edgecolors='red', s=50)

        # Common labels
        for ax in [ax0, ax1, ax2, ax3]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        plt.suptitle(f"3D Missing Data (ratio={missing_ratio*100:.1f}%): PCA vs. PPCA (Eigen) vs. PPCA (EM)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    missing_data()

