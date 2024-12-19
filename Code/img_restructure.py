from algos import MixtureOfPPCA
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import view_as_blocks, montage
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.util import view_as_blocks
from sklearn.decomposition import PCA
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.util import view_as_blocks
from sklearn.decomposition import PCA
from PIL import Image
from io import BytesIO
import requests
import os

def download_and_load_image(url, resize_shape=(256, 256), save_path=None):
    """Download an image from a URL, convert to grayscale, and resize."""
    try:
        print("Downloading image...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
        image = image.resize(resize_shape, Image.ANTIALIAS)
        if save_path:
            image.save(save_path)  # Save locally
        return img_as_float(np.array(image))
    except Exception as e:
        print(f"Failed to download image: {e}")
        print("Trying to load local fallback image...")
        fallback_path = "lena.png"  # Fallback local image path
        if os.path.exists(fallback_path):
            image = Image.open(fallback_path).convert('L').resize(resize_shape, Image.ANTIALIAS)
            return img_as_float(np.array(image))
        else:
            raise FileNotFoundError("Local fallback image not found. Please place 'lena.png' in the working directory.")


def split_image_into_blocks(image, block_size=(8, 8)):
    """Split an image into blocks of a given size."""
    padded_shape = (image.shape[0] // block_size[0] * block_size[0],
                    image.shape[1] // block_size[1] * block_size[1])
    image = image[:padded_shape[0], :padded_shape[1]]  # Ensure dimensions match
    blocks = view_as_blocks(image, block_size)
    return image, blocks, block_size


def reconstruct_image_from_blocks(reconstructed_blocks, original_shape, block_size):
    """Reconstruct an image from blocks."""
    reconstructed_image = np.block([[reconstructed_blocks[i, j]
                                     for j in range(reconstructed_blocks.shape[1])]
                                    for i in range(reconstructed_blocks.shape[0])])
    return reconstructed_image


if __name__ == "__main__":

    # Load the Lena image
    image = io.imread("Lenna_(test_image).png", as_gray=True)
    image = img_as_float(image)  # Normalize pixel values to [0, 1]
    # Display the original image
    plt.figure(figsize=(6, 6))
    plt.title("Original Lena Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

    # Split the image into blocks
    original_image, blocks, block_size = split_image_into_blocks(image, block_size=(8, 8))
    blocks_shape = blocks.shape
    flat_blocks = blocks.reshape(-1, block_size[0] * block_size[1])  # Flatten blocks

    print("Applying PCA globally to all blocks...")

    # Flatten all blocks: Each block is treated as a row (sample), flattened into a feature vector
    flat_blocks = blocks.reshape(-1, block_size[0] * block_size[1])

    # Fit PCA on all blocks together
    n_components = 5  # Choose the number of components to retain
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(flat_blocks)  # Transform all blocks
    pca_reconstructed_blocks = pca.inverse_transform(pca_transformed)  # Reconstruct blocks

    # Reshape reconstructed blocks back to the original block shape
    pca_reconstructed_blocks = pca_reconstructed_blocks.reshape(blocks_shape)

    # Apply MPPCA to each block
    print("Applying MPPCA...")
    mppca = MixtureOfPPCA(n_components=3, n_latent_dims=5, max_iter=50, tol=1e-4)
    mppca.fit(flat_blocks)
    mppca_reconstructed_blocks = []
    for block in flat_blocks:
        # Compute responsibilities
        responsibilities = mppca._e_step(block.reshape(1, -1)).flatten()
        
        # Find the most responsible component
        k = np.argmax(responsibilities)
        
        # Project into latent space and reconstruct
        W_k = mppca.latent_covars[k]  # Weight matrix for component k
        mean_k = mppca.means[k]
        block_centered = block - mean_k  # Center the block
        
        # Project to latent space and reconstruct back
        latent = W_k.T @ block_centered  # Projection
        reconstructed_block = W_k @ latent + mean_k
        
        mppca_reconstructed_blocks.append(reconstructed_block)

    # Convert reconstructed blocks to an array and reshape
    mppca_reconstructed_blocks = np.array(mppca_reconstructed_blocks).reshape(blocks_shape)

    # Reconstruct images
    pca_reconstructed_image = reconstruct_image_from_blocks(pca_reconstructed_blocks, original_image.shape, block_size)
    mppca_reconstructed_image = reconstruct_image_from_blocks(mppca_reconstructed_blocks, original_image.shape, block_size)

    # Display results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(original_image, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(pca_reconstructed_image, cmap="gray")
    ax[1].set_title("PCA Reconstructed Image")
    ax[1].axis("off")

    ax[2].imshow(mppca_reconstructed_image, cmap="gray")
    ax[2].set_title("MPPCA Reconstructed Image")
    ax[2].axis("off")

    plt.show()