from importdata_csv import *







#  The amount of variation explained as a function of the number of PCA components included,
#  the principal directions of the considered PCA components (either find a
# way to plot them or interpret them in terms of the features),
#  the data projected onto the considered principal components.


## exercise 2.1.6
import matplotlib.pyplot as plt
from scipy.linalg import svd

r = np.arange(1, X.shape[1] + 1)
plt.bar(r, np.std(X, 0))
plt.xticks(r, attributeNames[0:16])
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('NanoNose: attribute standard deviations')

## Investigate how standardization affects PCA

# Try this *later* (for last), and explain the effect
# X_s = X.copy() # Make a to be "scaled" version of X
# X_s[:, 2] = 100*X_s[:, 2] # Scale/multiply attribute C with a factor 100
# Use X_s instead of X to in the script below to see the difference.
# Does it affect the two columns in the plot equally?


# Subtract the mean from the data
Y1 = X - np.ones((N, 1)) * X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1)) * X.mean(0)
Y2 = Y2 * (1 / np.std(Y2, 0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=.4)
plt.title('NanoNose: Effect of standardization')
nrows = 3
ncols = 1

# Obtain the PCA solution by calculate the SVD of either Y1 or Y2
U, S, Vh = svd(Ys[0], full_matrices=False)
V = Vh.T  # For the direction of V to fit the convention in the course we transpose
# For visualization purposes, we flip the directionality of the
# principal directions such that the directions match for Y1 and Y2.
# if k == 1: V = -V; U = -U;

# Compute variance explained
rho = (S * S) / (S * S).sum()

# Compute the projection onto the principal components
Z = U * S;

# Plot projection
plt.subplot(nrows, ncols, 1)
C = len(classNames)
for c in range(C):
    plt.plot(Z[(y==c), i], Z[(y==c), j], '.', alpha=.5)
    plt.xlabel('PC' + str(i + 1))
    plt.ylabel('PC' + str(j + 1))
    plt.title(titles[0] + '\n' + 'Projection')
plt.legend(classNames)
plt.axis('equal')

# Plot attribute coefficients in principal component space
plt.subplot(nrows, ncols, 2)
for att in range(V.shape[1]):
    plt.arrow(0, 0, V[att, i], V[att, j])
    plt.text(V[att, i], V[att, j], attributeNames[att])
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xlabel('PC' + str(i + 1))
plt.ylabel('PC' + str(j + 1))
plt.grid()
# Add a unit circle
plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)),
         np.sin(np.arange(0, 2 * np.pi, 0.01)));
plt.title(titles[0] + '\n' + 'Attribute coefficients')
plt.axis('equal')

# Plot cumulative variance explained
plt.subplot(nrows, ncols, 3);
plt.plot(range(1, len(rho) + 1), rho, 'x-')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.title(titles[0] + '\n' + 'Variance explained')

plt.show()









