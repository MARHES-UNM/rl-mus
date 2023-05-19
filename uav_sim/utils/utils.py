import scipy.integrate
import scipy
import numpy as np
from math import sqrt, atan2


def cartesian2polar(point1=(0, 0), point2=(0, 0)):
    """Retuns conversion of cartesian to polar coordinates"""
    r = distance(point1, point2)
    alpha = angle(point1, point2)

    return r, alpha


def distance(point_1=(0, 0), point_2=(0, 0)):
    """Returns the distance between two points"""
    return sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)


def angle(point_1=(0, 0), point_2=(0, 0)):
    """Returns the angle between two points"""
    return atan2(point_2[1] - point_1[1], point_2[0] - point_1[0])


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u

    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # https://github.com/ssloy/tutorials/blob/master/tutorials/pendulum/lqr.py
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

    eig_vals, eig_vecs = np.linalg.eig(A - np.dot(B, K))

    return K, P, eig_vals

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
    
    x[k+1] = A x[k] + B u[k]
    
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    http://www.mwm.im/lqr-controllers-with-python/
    """
    #ref Bertsekas, p.151
    
    #first, try to solve the ricatti equation
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, P, eigVals
