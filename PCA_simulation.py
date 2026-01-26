# Import necessary libraries
import numpy as np # Numerical Computing
from sklearn.datasets import make_swiss_roll # Generate non-linear dataset
from sklearn.neighbors import kneighbors_graph # Build k-nearest neighbours graph (W matrix) 
from sklearn.decomposition import PCA # Standard PCA for comparison
import matplotlib.pyplot as plt # Plotting


# ====================================================================
# STEP 1: CREATE SWISS ROLL DATASET (Non-linear manifold)
# ====================================================================
n_samples = 1000 # Number of data points

X, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=0)
# make_swiss_roll returns 3D points (X) and a "color" parameter (intrinsic manifold coordinate)

# Take only 2 dimensions for visualization (project to 2D plane)
X = X[:, [0, 2]]  

# Center the data (subtract mean) - REQUIRED for PCA methods
X = X - X.mean(axis=0, keepdims=True)

# Create a simple binary class label: left side (x <= 0) vs right side (x > 0)
y = (X[:, 0] > 0).astype(int)  

# Should be matrix with 1000 points (rows) in 2 dimensions (coloumns)
print(f"Swiss Roll dataset: {X.shape}") 


# ====================================================================
# STEP 2: BUILD k-NEAREST NEIGHBOR GRAPH
# ====================================================================
k = 15 # Number of neighbors for each point

# Build adjacency matrix where A[i,j]=1 if j is among i's k nearest neighbors
A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)

# Make graph symmetric: if i is neighbor of j, j is also neighbor of i
A = 0.5 * (A + A.T)

# Convert sparse matrix A to dense array W for easier computation (Adjacency matrix to Similarty matrix)
W = A.toarray()  # W[i,j] = 1 if neighbours, 0 otherwise


# ====================================================================
# STEP 3: COMPUTE GRAPH LAPLACIAN MATRIX (L)
# ====================================================================

# Degree matrix: D[i,i] = sum of weights for node i 
D = np.diag(W.sum(axis=1))

# Graph Laplacian: L = D - W (measures smoothness on graph)
L = D - W


# ====================================================================
# STEP 4: GRAPH-LAPLACIAN PCA (Main Algorithm)
# ====================================================================

# Tranpose matrix to (d, n): d = dimension (2), n = number of samples (1000)
X_mat = X.T

# Compute X^T X (n × n sample similarity matrix), like a kernel/Gram matrix
XtX = X_mat.T @ X_mat # @ is matrix multiplication

# Regularization parameter: α balances PCA vs Laplacian terms
# α=0 → pure PCA, α→∞ → pure Laplacian embedding
alpha = 10.0

# Key equation: M = -X^T X + αL
# -X^T X: PCA term (negative because we minimize reconstruction error)
# +αL: Laplacian regularization term (penalizes mapping neighbors far apart)
M = -XtX + alpha * L

# Solve eigenvalue problem: M v = λ v
# eigh: for symmetric matrices, returns eigenvalues (ascending) and eigenvectors
eigvals, eigvecs = np.linalg.eigh(M)

# Sort indices from smallest to largest eigenvalue
idx = np.argsort(eigvals)


# ====================================================================
# STEP 4: SELECT EMBEDDING EIGENVECTOR 
# ====================================================================

# Choose the eigenvector that best separates the data
# Q contains the embedding coordinates (n x 1 matrix)
Q = eigvecs[:, idx[1:2]]  # Skip first (trivial), take second (best separator)

# Normalize eigenvectors to have unit length (orthonormal columns)
Q /= np.linalg.norm(Q, axis=0, keepdims=True)

# Compute projection matrix U = X Q (d(=2) x 1, like PCA loadings)
U = X_mat @ Q

# Compute 1D embedding: Z_LPCA = U^T X_mat -> (1 x 2) @ (2 x n) = (1 x n), then transpose to (n x 1)
Z_LPCA = (U.T @ X_mat).T

# Normalize Laplacian PCA coordinates for nicer visualization
Z_LPCA_normalized = (Z_LPCA - Z_LPCA.mean()) / Z_LPCA.std()


# ====================================================================
# STEP 5: STANDARD PCA (FOR COMPARISON)
# ====================================================================

pca = PCA(n_components=1) # Create PCA object, keep only 1 component
Z_PCA = pca.fit_transform(X) # Fit PCA and transform data to 1D

# Normalize PCA embedding for fair comparison
Z_PCA_normalized = (Z_PCA - Z_PCA.mean()) / Z_PCA.std()


# ====================================================================
# STEP 6: FIX SIGN AMBIGUITY FOR CONSISTENT VISUALIZATION
# ====================================================================
# Eigenvectors have sign ambiguity: v and -v are both valid
# We want red points (class 0) on LEFT (negative values), 
# blue points (class 1) on RIGHT (positive values)

# Check if red points have higher mean than blue points
if Z_PCA_normalized[y==0].mean() > Z_PCA_normalized[y==1].mean():
    # Flip sign to put reds on left
    Z_PCA_normalized = -Z_PCA_normalized

# Same for Laplacian PCA
if Z_LPCA_normalized[y==0].mean() > Z_LPCA_normalized[y==1].mean():
    Z_LPCA_normalized = -Z_LPCA_normalized


# ====================================================================
# STEP 7: VISUALIZE RESULTS
# ====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Create color list: red for class 0, blue for class 1
colors = ['red' if label == 0 else 'blue' for label in y]

# ----- PLOT 1: Plot 1: Original Swiss Roll (2D projection, colored by class) -----
axes[0].scatter(X[:, 0], X[:, 1], c=colors, s=10, alpha=0.7)
axes[0].set_title("Original Swiss Roll")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].grid(True, alpha=0.3)

#  ----- PLOT 2: STANDARD PCA RESULTS -----
# Compute class separation: difference between means of the two classes
sep_pca = abs(Z_PCA_normalized[y==1].mean() - Z_PCA_normalized[y==0].mean())

# Plot 1D embedding as points on a horizontal line
axes[1].scatter(Z_PCA_normalized, np.zeros_like(Z_PCA_normalized), 
               c=colors, s=10, alpha=0.7)
axes[1].set_title(f"Standard PCA\nClass Separation: {sep_pca:.3f}")
axes[1].set_xlabel("PC1 (normalized)")
axes[1].set_yticks([]) # Hide y-axis (it's just 0)
axes[1].grid(True, alpha=0.3)

# ----- PLOT 3: LAPLACIAN PCA RESULTS -----
sep_lpca = abs(Z_LPCA_normalized[y==1].mean() - Z_LPCA_normalized[y==0].mean())
axes[2].scatter(Z_LPCA_normalized, np.zeros_like(Z_LPCA_normalized), 
               c=colors, s=10, alpha=0.7)
axes[2].set_title(f"Laplacian PCA (α={alpha})\nClass Separation: {sep_lpca:.3f}")
axes[2].set_xlabel("Embedding (normalized)")
axes[2].set_yticks([])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ====================================================================
# STEP 8: QUANTITATIVE EVALUATION
# ====================================================================
print("\n" + "="*60)
print("QUANTITATIVE RESULTS")
print("="*60)
print(f"Standard PCA:")
print(f"  Class 0 mean: {Z_PCA_normalized[y==0].mean():.3f}")
print(f"  Class 1 mean: {Z_PCA_normalized[y==1].mean():.3f}")
print(f"  Separation: {sep_pca:.3f}")

print(f"\nLaplacian PCA:")
print(f"  Class 0 mean: {Z_LPCA_normalized[y==0].mean():.3f}")
print(f"  Class 1 mean: {Z_LPCA_normalized[y==1].mean():.3f}")
print(f"  Separation: {sep_lpca:.3f}")

# Calculate improvement factor
print(f"\nImprovement: {sep_lpca/sep_pca:.1f}x better separation!")