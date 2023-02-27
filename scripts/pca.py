from importdata_csv import *
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
X = X - np.ones((N, 1)) * X.mean(0)
X = X * (1 / np.std(X, 0))

# Obtain the PCA solution by calculate the SVD of X
U, S, Vh = svd(X, full_matrices=False)
V = Vh.T  # For the direction of V to fit the convention in the course we transpose

# Compute variance explained
rho = (S * S) / (S * S).sum()
print(rho)
# Compute the projection onto the principal components
Z = U * S
print(Z.shape)

# Plot the explained variance of principal components
plt.figure(figsize=(10, 15))
threshold = 0.9
plt.plot(range(1, len(rho) + 1), rho, 'x-')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()

# Based on the above figure, we see that 6 principal components
# can explain over 90% of the variance in the data.

NoPC = 6

# Plot coefficients of the principal components.
plt.figure(figsize=(10, 15))
pcs = np.arange(NoPC)
legendStrs = ['PC'+str(e+1) for e in pcs]
# c = ['r', 'g', 'b', 'c', 'k', 'y']
bw = 0.1
r = np.arange(1, M+1)
for i in pcs:
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+3*bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')

NoPC = 3

# Plot the projected scatter matrix
r_mask = np.ones(N, dtype=bool)
# r_mask = np.random.uniform(size=N) < 0.33
plt.figure(figsize=(15, 10))
C = len(classNames)
for i in range(NoPC):
    for j in range(NoPC):
        for c in range(C):
            class_mask = (y == c) & r_mask
            plt.subplot(NoPC, NoPC, NoPC*i+j+1)
            plt.scatter(Z[class_mask, j], Z[class_mask, i], s=4, c=class_colors[c])
            if j == 0:
                plt.ylabel('PC' + str(i + 1))
            else:
                plt.yticks([])
            if i == NoPC-1:
                plt.xlabel('PC' + str(j + 1))
            else:
                plt.xticks([])
            # plt.title(titles + '\n' + 'Projection')
plt.legend(classNames, bbox_to_anchor=(1.04, 0.5), loc="center left")
# plt.axis('equal')

plt.show()
