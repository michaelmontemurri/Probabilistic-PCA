import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, transform
from skimage.util import view_as_windows 
from algos import PPCAEigen, PPCAEM, MixtureOfPPCA

def extract_patches_2d(image, patch_size=(8, 8), step=8):
    """
    Partition a 2D image into (potentially overlapping) patches.

    Args:
        image (ndarray): 2D grayscale image, shape (H, W).
        patch_size (tuple): (patch_height, patch_width).
        step (int): the stride or step between consecutive patches.

    Returns:
        patches (ndarray): shape (N, patch_height*patch_width).
        patch_coords (list): list of (row, col) top-left coords for each patch.
    """
    patches = []
    coords = []
    pH, pW = patch_size
    H, W = image.shape

    for r in range(0, H - pH + 1, step):
        for c in range(0, W - pW + 1, step):
            patch = image[r:r+pH, c:c+pW]
            patches.append(patch.flatten())
            coords.append((r, c))
    patches = np.array(patches)  # (N, pH*pW)
    return patches, coords

def reconstruct_image_from_patches(patches, coords, image_shape,
                                   patch_size=(8, 8), step=8):
    """
    Naive reassembly of an image from its patches (no overlap or avg in overlapping regions).
    If step < patch_size, it will average in overlapped regions.

    Args:
        patches (ndarray): shape (N, pH*pW), flattened patches.
        coords (list): list of (row, col) top-left coords for each patch.
        image_shape (tuple): (H, W).
        patch_size (tuple): (pH, pW).
        step (int): stride or step used in extraction.

    Returns:
        reconst (ndarray): reassembled 2D image, shape (H, W).
    """
    pH, pW = patch_size
    H, W = image_shape
    reconst = np.zeros((H, W))
    weight = np.zeros((H, W))

    for patch_vec, (r, c) in zip(patches, coords):
        patch_2d = patch_vec.reshape(pH, pW)
        reconst[r:r+pH, c:c+pW] += patch_2d
        weight[r:r+pH, c:c+pW] += 1.0

    # average in overlapping areas
    valid_mask = weight > 0
    reconst[valid_mask] /= weight[valid_mask]
    return reconst

def image_reconstruction():
    # Load and preprocess a grayscale image
    cat_rgb = data.chelsea()  # shape approx (300,451,3)
    cat_gray = color.rgb2gray(cat_rgb)  # shape ~ (300,451)
    cat_gray = transform.resize(cat_gray, (128,128))  # smaller for speed

    # Extract patches
    patch_size = (8, 8)
    step = 8  # non-overlapping
    patches, coords = extract_patches_2d(cat_gray, patch_size=patch_size, step=step)
    print(f"Extracted {len(patches)} patches from image of size {cat_gray.shape}.")

    # Train each model and reconstruct patches
    q = 10  # latent dimension

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=q, random_state=42)
    pca.fit(patches)
    patches_pca_rec = pca.inverse_transform(pca.transform(patches))

    # PPCA (Eigen)
    ppca_eigen = PPCAEigen(q=q, max_iter=50, tol=1e-4, random_state=42)
    ppca_eigen.fit(patches)
    patches_ppca_eig_rec = ppca_eigen.reconstruct(patches)

    # PPCA (EM)
    ppca_em = PPCAEM(q=q, max_iter=50, tol=1e-4, random_state=42)
    ppca_em.fit(patches)
    patches_ppca_em_rec = ppca_em.reconstruct(patches)

    # MPPCA
    mppca = MixtureOfPPCA(n_components=4, q=q, max_iter=30, tol=1e-4, random_state=42)
    mppca.fit(patches)
    patches_mppca_rec = mppca.reconstruct(patches)

    # Reassemble the images
    cat_pca_rec = reconstruct_image_from_patches(patches_pca_rec, coords,
                                                 cat_gray.shape,
                                                 patch_size=patch_size, step=step)
    cat_ppca_eig_rec = reconstruct_image_from_patches(patches_ppca_eig_rec, coords,
                                                      cat_gray.shape,
                                                      patch_size=patch_size, step=step)
    cat_ppca_em_rec = reconstruct_image_from_patches(patches_ppca_em_rec, coords,
                                                     cat_gray.shape,
                                                     patch_size=patch_size, step=step)
    cat_mppca_rec = reconstruct_image_from_patches(patches_mppca_rec, coords,
                                                   cat_gray.shape,
                                                   patch_size=patch_size, step=step)

    # Visualize full images + a zoomed-in region
    fig, axs = plt.subplots(2, 5, figsize=(16, 6))

    # Top row: entire image comparisons
    axs[0, 0].imshow(cat_gray, cmap='gray')
    axs[0, 0].set_title("Original")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(cat_pca_rec, cmap='gray')
    axs[0, 1].set_title("PCA")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(cat_ppca_eig_rec, cmap='gray')
    axs[0, 2].set_title("PPCA-Eigen")
    axs[0, 2].axis("off")

    axs[0, 3].imshow(cat_ppca_em_rec, cmap='gray')
    axs[0, 3].set_title("PPCA-EM")
    axs[0, 3].axis("off")

    axs[0, 4].imshow(cat_mppca_rec, cmap='gray')
    axs[0, 4].set_title("MPPCA")
    axs[0, 4].axis("off")

    # Zoom region: let's define a small sub-box
    r0, r1 = 30, 70
    c0, c1 = 20, 80

    zoom_orig = cat_gray[r0:r1, c0:c1]
    zoom_pca = cat_pca_rec[r0:r1, c0:c1]
    zoom_eig = cat_ppca_eig_rec[r0:r1, c0:c1]
    zoom_em  = cat_ppca_em_rec[r0:r1, c0:c1]
    zoom_mppca = cat_mppca_rec[r0:r1, c0:c1]

    # Bottom row: zoomed region
    axs[1, 0].imshow(zoom_orig, cmap='gray')
    axs[1, 0].set_title("Orig (Zoom)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(zoom_pca, cmap='gray')
    axs[1, 1].set_title("PCA (Zoom)")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(zoom_eig, cmap='gray')
    axs[1, 2].set_title("PPCA-Eigen (Zoom)")
    axs[1, 2].axis("off")

    axs[1, 3].imshow(zoom_em, cmap='gray')
    axs[1, 3].set_title("PPCA-EM (Zoom)")
    axs[1, 3].axis("off")

    axs[1, 4].imshow(zoom_mppca, cmap='gray')
    axs[1, 4].set_title("MPPCA (Zoom)")
    axs[1, 4].axis("off")

    plt.suptitle("Image Patch Reconstruction: Full View + Zoomed Region", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_reconstruction()
