import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
from pylbfgs import owlqn
import zhang

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def evaluate(x, g, step):
  """An in-memory evaluation callback."""

  # we want to return two things: 
  # (1) the norm squared of the residuals, sum((Ax-b).^2), and
  # (2) the gradient 2*A'(Ax-b)

  # expand x columns-first
  x2 = x.reshape((nx, ny)).T

  # Ax is just the inverse 2D dct of x2
  Ax2 = idct2(x2)

  # stack columns and extract samples
  Ax = Ax2.T.flat[ri].reshape(b.shape)

  # calculate the residual Ax-b and its 2-norm squared
  Axb = Ax - b
  fx = np.sum(np.power(Axb, 2))

  # project residual vector (k x 1) onto blank image (ny x nx)
  Axb2 = np.zeros(x2.shape)
  Axb2.T.flat[ri] = Axb # fill columns-first

  # A'(Ax-b) is just the 2D dct of Axb2
  AtAxb2 = 2 * dct2(Axb2)
  AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

  # copy over the gradient vector
  np.copyto(g, AtAxb)

  return fx

def progress(x, g, fx, xnorm, gnorm, step, k, ls):
  """Just display the current iteration.
  """
  print('Iteration {}'.format(k))
  return 0



path = '/src/zhang/demo/imgs/rooster.jpg'
sample_scale = 0.1
zoom_scale = 1

# read original image and downsize for speed
Xorig = spimg.imread(path, flatten=True, mode='L') # read in grayscale
X = spimg.zoom(Xorig, zoom_scale)
# plt.imshow(X, cmap='gray')
print('X')
print(X.shape)
ny,nx = X.shape

# extract small sample of signal
k = round(nx * ny * sample_scale) # 50% sample
if k < 1:
  print('Sample size too small')

ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
b = X.T.flat[ri]
b = np.expand_dims(b, axis=1)

print('b')
print(b.shape)

# # create dct matrix operator using kron (memory errors for large ny*nx)
# A = np.kron(
#     spfft.idct(np.identity(nx), norm='ortho', axis=0),
#     spfft.idct(np.identity(ny), norm='ortho', axis=0)
#     )
# A = A[ri,:] # same as phi times kron
# print('A')
# print(A.shape)

# # do L1 optimization
# vx = cvx.Variable((nx * ny, 1))
# print('vx')
# print(vx.shape)
# objective = cvx.Minimize(cvx.norm(vx, 1))

# constraints = [A*vx == b]
# prob = cvx.Problem(objective, constraints)
# result = prob.solve(verbose=True)
# Xat2 = np.array(vx.value).squeeze()

Xat2 = owlqn(nx*ny, evaluate, progress, 5)

# reconstruct signal
Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct2(Xat)

# confirm solution
if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
    print('Warning: values at sample indices don\'t match original.')

# create images of mask (for visualization)
mask = np.zeros(X.shape)
mask.T.flat[ri] = 255
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]

plt.imshow(Xa, cmap='gray')
plt.show()

Xa_col = zhang.colorize_from_grayscale(Xa)
plt.imshow(Xa_col)
plt.show()
