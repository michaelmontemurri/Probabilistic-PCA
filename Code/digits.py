import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from algos import MixtureOfPPCA

# use regular pca to denoise the data
def regular_pca_denoise(X_noisy, n_components):
    pca = PCA(n_components=n_components)
    X_projected = pca.fit_transform(X_noisy)  # reduce dimensions
    X_reconstructed = pca.inverse_transform(X_projected)  # map back to original space
    return X_reconstructed

# plot original, noisy, pca, and mppca reconstructions for comparison
def plot_comparison(original, noisy, regular, mppca, n=10):
    plt.figure(figsize=(20, 8))
    for i in range(n):
        # show original images
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original", fontsize=12)

        # show noisy images
        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(noisy[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Noisy", fontsize=12)

        # show pca denoised images
        ax = plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow(regular[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("PCA", fontsize=12)

        # show mppca denoised images
        ax = plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(mppca[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("MPPCA", fontsize=12)

    plt.show()

# visualize data clusters with labels
def visualize_clusters(data, labels, title="MPPCA Clusters"):
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        cluster_data = data[labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {label}", alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.show()

# calculate reconstruction error between original and reconstructed data
def reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2, axis=1)

# process one digit: train models, denoise, and calculate errors
def process_digit(digit, X_train, X_noisy, y_train, n_components, n_clusters):
    print(f"training models for digit {digit}...")
    X_train_digit = X_train[y_train == digit]
    X_noisy_digit = X_noisy[y_train == digit]

    # apply regular pca
    print(f"applying regular pca for digit {digit}...")
    X_regular_denoised = regular_pca_denoise(X_noisy_digit, n_components)

    # apply mppca
    print(f"applying mppca for digit {digit}...")
    model = MixtureOfPPCA(n_components=n_clusters, n_latent_dims=n_components)
    model.fit(X_noisy_digit)

    # reconstruct mppca denoised images
    X_mppca_denoised = np.zeros_like(X_noisy_digit)
    responsibilities = model._e_step(X_noisy_digit)
    for k in range(n_clusters):
        cluster_mean = model.means[k]
        eigenvalues, eigenvectors = np.linalg.eigh(model.covars[k])
        top_eigenvectors = eigenvectors[:, -n_components:]
        cluster_data = X_noisy_digit - cluster_mean
        projections = cluster_data @ top_eigenvectors
        reconstructions = projections @ top_eigenvectors.T + cluster_mean
        X_mppca_denoised += responsibilities[:, k][:, np.newaxis] * reconstructions

    # calculate reconstruction errors
    regular_error = reconstruction_error(X_train_digit, X_regular_denoised)
    mppca_error = reconstruction_error(X_train_digit, X_mppca_denoised)

    return {
        "digit": digit,
        "regular_pca": X_regular_denoised,
        "mppca": X_mppca_denoised,
        "original": X_train_digit,
        "noisy": X_noisy_digit,
        "regular_error": regular_error,
        "mppca_error": mppca_error,
    }

if __name__ == "__main__":
    # load mnist dataset
    print("loading dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)

    # preprocess: scale data and split into train/test
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    # reduce dataset size for debugging
    selected_digits = [8]  # focus on a single digit for now
    X_train = X_train[np.isin(y_train, selected_digits)]
    y_train = y_train[np.isin(y_train, selected_digits)]
    X_train = X_train[:500]  # limit the number of samples
    y_train = y_train[:500]

    # add gaussian noise to images
    noise_level = 0.5
    X_noisy = X_train + noise_level * np.random.normal(size=X_train.shape)

    # set parameters for testing
    n_components = 10
    n_clusters = 10

    # process each digit
    models = {}
    for digit in selected_digits:
        result = process_digit(digit, X_train, X_noisy, y_train, n_components, n_clusters)
        models[digit] = result

    # plot results for one digit
    chosen_digit = 8
    print(f"plotting comparison for digit {chosen_digit}...")
    plot_comparison(
        models[chosen_digit]["original"],
        models[chosen_digit]["noisy"],
        models[chosen_digit]["regular_pca"],
        models[chosen_digit]["mppca"]
    )

    # visualize reconstruction errors
    print("visualizing reconstruction error distribution...")
    plt.figure(figsize=(8, 6))
    plt.hist(models[chosen_digit]["regular_error"], bins=50, alpha=0.5, label="PCA Error")
    plt.hist(models[chosen_digit]["mppca_error"], bins=50, alpha=0.5, label="MPPCA Error")
    plt.title("Reconstruction Error Distribution for Digit 8")
    plt.legend()
    plt.show()

    # visualize clusters for the chosen digit
    print(f"visualizing mppca clusters for digit {chosen_digit}...")
    mppca_labels = np.argmax(models[chosen_digit]["regular_error"], axis=0)
    visualize_clusters(models[chosen_digit]["noisy"], mppca_labels, title="MPPCA Clusters for Digit 8")
