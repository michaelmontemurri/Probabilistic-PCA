import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from algos import MixtureOfPPCA

# generate synthetic helix-shaped data
def generate_helix(n_points, noise_level=0.1, z_noise=0.5):
    t = np.linspace(0, 4 * np.pi, n_points)  # parametric variable
    x = np.cos(t)  # x-coordinates of helix
    y = np.sin(t)  # y-coordinates of helix
    z = t + z_noise * np.random.normal(size=n_points)  # z-coordinates with added noise
    helix = np.stack([x, y, z], axis=1)  # combine into (x, y, z)
    helix += noise_level * np.random.normal(size=helix.shape)  # add noise to all dimensions
    return helix

# visualize clustering results in 3D
def visualize_clusters(X, labels, title, log_likelihood, means=None):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')  # 3D scatter plot
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=20, alpha=0.8)
    if means is not None:  # optionally show cluster centers
        ax.scatter(means[:, 0], means[:, 1], means[:, 2], c='red', s=100, marker='X', label='Cluster Centers')
    ax.set_title(f"{title}\nLog-Likelihood per Data Point: {log_likelihood:.4f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    return fig

if __name__ == "__main__":
    # generate synthetic helix data
    print("generating helical data...")
    n_points = 1000
    X = generate_helix(n_points, noise_level=0.15, z_noise=0.5)

    # apply gaussian mixture model (gmm)
    print("applying gaussian mixture model...")
    n_components = 10  # number of clusters
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)  # fit gmm to the data
    gmm_labels = gmm.predict(X)  # get cluster labels
    gmm_log_likelihood = gmm.score(X)  # compute average log-likelihood per data point

    # apply mixture of ppca (mppca)
    print("applying mixture of ppca...")
    mppca = MixtureOfPPCA(n_components=n_components, n_latent_dims=2)  # 2 latent dimensions
    mppca.fit(X)  # fit mppca to the data
    mppca_responsibilities = mppca._e_step(X)  # compute responsibilities
    mppca_labels = np.argmax(mppca_responsibilities, axis=1)  # assign data to most likely cluster
    mppca_log_likelihood = mppca.log_likelihoods[-1]  # final log-likelihood

    # visualize gmm and mppca results
    print("visualizing results...")
    fig1 = visualize_clusters(X, gmm_labels, "Gaussian Mixture Model", gmm_log_likelihood, means=gmm.means_)
    fig2 = visualize_clusters(X, mppca_labels, "Mixture of PPCA", mppca_log_likelihood, means=mppca.means)
    plt.show()
