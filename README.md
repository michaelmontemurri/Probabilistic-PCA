# Probabilistic PCA and Kernel PCA
This repository contains our project on implementing and exploring **Probabilistic PCA (PPCA)**, **Mixtures of Probabilistic PCA (MPPCA)**, and **Kernel PCA**, along with their applications. The project also includes a Bayesian optimization approach for hyperparameter selection in Kernel PCA.

---

## Repository Structure
```plaintext

├── Code/
│   ├── notebooks/
│   │   ├── demos.ipynb           # Walkthrough of demos for PPCA (Eigen), PPCA (EM), and MPPCA
│   │   ├── kernel_pca.ipynb      # Examples with Kernel PCA and Bayesian hyperparameter optimization
│   ├── scripts/
│       ├── algos.py              # Implementation of PPCA (Eigen), PPCA (EM), and MPPCA
│       ├── clusters.py           # Demonstration of MPPCA vs PCA
│       ├── img_reconstruct.py    # Demonstration of image reconstruction
│       ├── missing_data.py       # Demonstration of handling missing data with PPCA
│       ├── ppca_comparison.py    # Demonstration of computational comparison between EM and Eig PPCA
├── Papers/
│   ├── relevant papers    

├── Reports/
│   ├── final report and poster        
└── README.md
```
## Project Overview

### Goals

-Implement and compare PPCA (Eigen), PPCA (EM), and MPPCA for handling high-dimensional data.

-Demonstrate the use of MPPCA for better reconstruction and handling of heterogeneous data.

-Explore Kernel PCA for capturing nonlinear structures in data.

-Develop a Bayesian optimization approach for selecting hyperparameters in Kernel PCA.

## Notebooks

`demos.ipynb`

Walkthrough of various demos illustrating:

-PPCA (Eigen): Eigen decomposition-based approach.

-PPCA (EM): Expectation-Maximization algorithm appraoch.

-MPPCA: Mixture model for local subspace learning.

Includes examples like missing data reconstruction and image patch modeling.

`kernel_pca.ipynb`

Examples showcasing the application of Kernel PCA for nonlinear data.

Includes our Bayesian optimization approach for hyperparameter selection.


## Implementations
`scripts/algos.py` contains core implementations of PPCA (Eigen), PPCA (EM), and MPPCA

## References
All relevant papers references are stored in the `Papers` folder

