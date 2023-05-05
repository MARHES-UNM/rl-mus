import scipy.integrate
import scipy
import numpy as np

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u

    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # https://github.com/ssloy/tutorials/blob/master/tutorials/pendulum/lqr.py
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = np.dot(np.linalg.inv(R), np.dot(B.T, X))

    eig_vals, eig_vecs = np.linalg.eig(A - np.dot(B, K))

    return K, X, eig_vals