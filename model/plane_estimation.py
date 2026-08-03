import numpy as np
from scipy.integrate import dblquad
from scipy.linalg import eigh
import matplotlib.pyplot as plt

"""
Implementasion of a recurrent Bingham distribution filter using Kurz et al. 2014 
see references. 

"""

class BinghamDistribution:
    """
    Represents a single 2D Bingham distribution.
    
    M : (3,3) orthogonal matrix
        Defines the orientation of the distribution in space.
        The last column is the mode, the most probable plane normal.
        Columns must be perpendicular unit vectors.
    
    Z : (3,3) diagonal matrix
        Defines the concentration, how certain the distribution is.
        Z = diag(z1, z2, 0) where z1 <= z2 <= 0 always.
        z1 : concentration along m1 
        z2 : concentration along m2 
        The third is fixed at 0
        The more negative it is the more certain the estimation is, 
        and the smaller spread of the probability distribution. And opposite
        the closer to 0 the more uncertain.
        
        
    """
    def __init__(self, M, Z):
        M = np.array(M, dtype=float) 
        Z = np.array(Z, dtype=float)

        self.M = M
        self.Z = Z
        self.z1 = Z[0, 0]
        self.z2 = Z[1, 1]


    @property
    def mode(self):
        return self.M[:, -1] #Returns the mode, most probable plane normal direction

    @property
    def A(self):
        """The distribution in its exponent form, exp(x^T A x) / F.

        M and Z are the eigendecomposition of A, so this is the same object seen
        from the other side; it is what composes under multiplication.
        """
        return self.M @ self.Z @ self.M.T

    def pdf(self, x):
        """
        The probability density function of the Bingham function. Taken from the paper. 
        Uses M, Z and F 
        Outputs: a probability density
        """
        x = np.asarray(x, dtype=float) 
        F = compute_F(self.z1, self.z2) #normalisation constant
        return np.exp(x @ self.A @ x) / F

    #TODO: also more debugging can remove this later
    def __repr__(self):
        return (f"BinghamDistribution(\n"
                f"  M={self.M},\n"
                f"  Z={self.Z},\n"
                f"  mode={self.mode}\n"
                f"  z1={self.z1:.4f}, z2={self.z2:.4f}\n)")
        
def _integrand(phi, t, z1, z2):
    """
    Integrand for the Bingham normalisation constant on S².
    """
    return np.exp(z1 * (1 - t**2) * np.cos(phi)**2
                + z2 * (1 - t**2) * np.sin(phi)**2)

def compute_F(z1, z2):
    """
    Computes the normalisation constant.
    with Z = diag(z1, z2, 0).
    
    Used to analyse results, during experiemnts we will use lookup tables.
    Beacuse of the computasional overhead of computing confluent hypergeometric function.
    
    
    Uses z1, z2 - concetration parameters
    """
    #Calculates the double integral, with limits -1,1 for the outer integral over t
    # And the inner integral over phi
    val, _ = dblquad(_integrand, -1, 1, 0, 2*np.pi, args=(z1, z2))
    return val



def _bingham_from_A(A):
    """Recover (M, Z) from an exponent matrix, enforcing the z3 = 0 convention.

    eigh returns eigenvalues ascending, so subtracting the last one shifts the
    largest to zero. That shift is a constant in the exponent and so is absorbed
    by the normalisation, which is why it is free to impose.
    """
    eigenvalues, eigenvectors = eigh(A)
    return BinghamDistribution(eigenvectors, np.diag(eigenvalues - eigenvalues[-1]))


def multiply_bingham(b1, b2):
    """
    The bayesian update opperation. Since Bingham probability PDF´s is closed under multiplication,
    this return an exact with no approximation. 
    
    FORMULA:
    Equation (12)
    """
    #From equation (12), multiplying two Bingham densities adds their exponents
    return _bingham_from_A(b1.A + b2.A)


def predict(estimate,  alpha):
    """
    Does an approximation of isotropic diffusion on S^2, meaning spread the distribution outward uniformly.
    This is a departue and simplification of the kurz paper
    """
    #Add a slight uncertinity to the plane normal estimation
    Z_pred = alpha * estimate.Z
    # Last diagonal entry has to be 0
    Z_pred[2, 2] = 0.0
    #M left unchanged, new more uncertain Z
    return BinghamDistribution(estimate.M.copy(), Z_pred)


def update(prediction, measurement, kappa):
    """
    This is the Bayesian update step.
    Corrects the prediction using a new measurement.
    """
    A_likelihood = -kappa * np.outer(measurement, measurement)
    #Bayes rule, + becasue it is applied to distributions inside an exponent
    return _bingham_from_A(prediction.A + A_likelihood)


def run_bingham_filter(initial_estimate, measurements, kappa, alpha=0.999):
    """
    This is the complete reccursive Bingham distribution filter. 
    
    Predict then update, repeat. 
    """
    estimates = []
    current = initial_estimate
    #loop over each displacement vector
    for displacement in measurements:
        predicted = predict(current, alpha)      # mode fixed, Z deflated
        current = update(predicted, displacement, kappa)  # accumulate displacement evidence
        estimates.append(current)
 
    return estimates
 


def uniform_prior():
    """Uniform distribution on S². Maximum uncertainty about the plane normal."""
    return BinghamDistribution(np.eye(3), np.zeros((3, 3)))

def plane_axes(n):
    """
    Returns two orthonormal vectors e1, e2 that span the plane
    perpendicular to unit vector n. As well as unit vector n
    """
    n   = np.asarray(n, dtype=float)
    n  /= np.linalg.norm(n)
    
    ## Gram-Schmidt: build two axes perpendicular to n
    arb = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = arb - (arb @ n) * n
    e1 /= np.linalg.norm(e1)
    e2  = np.cross(n, e1)
    return e1, e2, n



def init_from_normal_guess(n_hat_guess, z1=-0.1, z2=-0.1):
    """
    Weak prior centred on a guessed plane normal.

    Builds an orthonormal frame with n_hat_guess as the mode.
    z1, z2 small negative means weak prior. More negative meanse more certain.
    """

    e1, e2, n_normalised = plane_axes(n_hat_guess)

    M = np.column_stack([e1, e2, n_normalised])  # last column is the mode
    Z = np.diag([z1, z2, 0.0])
    return BinghamDistribution(M, Z)


#TODO not working
def flip_mode(b: BinghamDistribution) -> BinghamDistribution:
    """
    Flip the mode of a Bingham distribution to its antipode.
    """
    M_flipped = b.M.copy()
    M_flipped[:, -1] = -M_flipped[:, -1]
    return BinghamDistribution(M_flipped, b.Z.copy())


def bingham_pdf_on_sphere(estimate, F, n_points=100):
    """
    Visulaisation helper
    """
    theta = np.linspace(0, np.pi,    n_points) #polar angles
    phi   = np.linspace(0, 2*np.pi, n_points) #azimuthal angles
    T, P  = np.meshgrid(theta, phi) #tile the sphere manifold with all possible pairings

    #spherical coordinates to cartesian conversion
    X = np.sin(T) * np.cos(P)
    Y = np.sin(T) * np.sin(P)
    Z = np.cos(T)

    pts  = np.stack([X, Y, Z], axis=-1)
    vals = np.einsum('ijk,kl,ijl->ij', pts, estimate.A, pts)
    PDF  = np.exp(vals) / F

    return X, Y, Z, PDF

def angular_error_s2(est_mode, n_true):
    """ Use for visualisation and checking in experiemnts """
    dot = np.clip(abs(np.dot(est_mode, n_true)), 0.0, 1.0)
    return np.rad2deg(np.arccos(dot))

def plot_bingham_convergence(n_hat_hist, gap_hist, n_true,
                             title="",
                             figsize=(8, 5.5)):
    """
    Plot the recurrent Bingham plane filter over a run.
    """

    n_hat_hist = np.asarray(n_hat_hist, dtype=float)
    gap_hist   = np.asarray(gap_hist,   dtype=float)
    n_true     = np.asarray(n_true,     dtype=float)
    n_true     = n_true / np.linalg.norm(n_true)

    err   = np.array([angular_error_s2(n, n_true) for n in n_hat_hist])
    steps = np.arange(1, len(err) + 1)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(steps, err, color="#d62728", lw=2)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set_ylabel("angular error (deg)"); axes[0].set_ylim(bottom=0)
    axes[0].set_title(title, pad=22)          # pad gives the new top axis room
    axes[0].grid(True, alpha=0.3, ls="--")

    # mirrored time axis along the top of the upper panel
    secax = axes[0].secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    secax.set_xlabel("time step")

    axes[1].plot(steps, gap_hist, color="navy", lw=1.5)
    axes[1].set_ylabel("certainty  z₂−z₁"); axes[1].set_xlabel("time step")
    axes[1].set_title("Filter concentration (larger gap = more committed)")
    axes[1].grid(True, alpha=0.3, ls="--")

    fig.tight_layout()
    return fig, axes