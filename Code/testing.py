import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
from matplotlib import cm  # for color mapping
from algos import MixtureOfPPCA

if __name__ == "__main__":
    # set parameters
    n_samples = 700  # total number of data points
    n_features = 3  # number of dimensions
    n_components = 5  # number of clusters
    n_latent_dims = 2  # number of latent dimensions

    # generate synthetic 3d data with ellipsoidal clusters
    def generate_random_covariance(n_features):
        # create a random positive-definite covariance matrix
        A = np.random.rand(n_features, n_features)
        return np.dot(A, A.T)  # symmetric positive-definite matrix

    # set up cluster means and covariances
    cluster_means = [np.random.rand(n_features) * 10 for _ in range(n_components)]
    cluster_covars = [generate_random_covariance(n_features) for _ in range(n_components)]

    # sample data points from each cluster
    X = np.vstack([
        np.random.multivariate_normal(cluster_means[k], cluster_covars[k], size=n_samples // n_components)
        for k in range(n_components)
    ])

    # fit mppca to the data
    model = MixtureOfPPCA(n_components=n_components, n_latent_dims=n_latent_dims)
    model.fit(X)  # train the model
    responsibilities = model.predict(X)  # get responsibilities (soft cluster assignments)
    labels = np.argmax(responsibilities, axis=1)  # assign to the most likely cluster

    # create an interactive 3d plot
    fig = go.Figure()
    colors = cm.tab10.colors[:n_components]  # use distinct colors for each cluster

    for k in range(n_components):
        # plot data points for each cluster
        cluster_data = X[labels == k]
        fig.add_trace(go.Scatter3d(
            x=cluster_data[:, 0], y=cluster_data[:, 1], z=cluster_data[:, 2],
            mode='markers', marker=dict(size=5, color=f'rgb{colors[k]}', opacity=0.8),
            name=f"Cluster {k+1}"
        ))
        # plot the top principal component as a line
        mean_k = model.means[k]
        eigenvalues, eigenvectors = np.linalg.eigh(model.covars[k])
        top_eigenvector = eigenvectors[:, -1]  # largest eigenvector
        line_start = mean_k - 5 * top_eigenvector  # start point of the line
        line_end = mean_k + 5 * top_eigenvector  # end point of the line
        fig.add_trace(go.Scatter3d(
            x=[line_start[0], line_end[0]], y=[line_start[1], line_end[1]], z=[line_start[2], line_end[2]],
            mode='lines', line=dict(color=f'rgb{colors[k]}', width=5), name=f"PC Cluster {k+1}"
        ))

    # set up the plot layout
    fig.update_layout(
        title="3D Mixture of PPCA with Local Principal Components",
        scene=dict(
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            zaxis_title="Feature 3",
        )
    )
    fig.show()
