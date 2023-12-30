import subprocess
import scipy.integrate
import scipy
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, atan2


def cir_traj(t, e=0.5, r=1, x_c=0, y_c=0):
    # https://web2.qatar.cmu.edu/~gdicaro/16311-Fall17/slides/control-theory-for-robotics.pdf
    """Generate circular trajectory
       https://ieeexplore-ieee-org.libproxy.unm.edu/stamp/stamp.jsp?tp=&arnumber=911382

    Args:
        t (_type_): _description_
        e (float, optional): _description_. Defaults to 0.5.
        r (int, optional): _description_. Defaults to 1.
        x_c (int, optional): _description_. Defaults to 0.
        y_c (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    x_r = x_c + r * np.cos(e * t)
    y_r = y_c + r * np.sin(e * t)
    # x_r = np.sin(t / 10)
    # y_r = np.sin(e*t)
    theta_r = e * t
    v_r = e * r
    w_r = e
    return np.array([x_r, y_r, theta_r, v_r, w_r])


def get_git_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


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


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    http://www.mwm.im/lqr-controllers-with-python/
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * P * B + R) * (B.T * P * A))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, P, eigVals


def plot_traj(uav_des_traj, uav_trajectory, title=""):
    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(411)
    ax.plot(uav_des_traj[:, 0])
    ax.plot(uav_trajectory[:, 0])
    ax.set_xlabel("t(s)")
    ax.set_ylabel("x (m)")

    ax1 = fig.add_subplot(412)
    ax1.plot(uav_des_traj[:, 1])
    ax1.plot(uav_trajectory[:, 1])
    ax1.set_ylabel("y (m)")

    ax2 = fig.add_subplot(413)
    ax2.plot(uav_des_traj[:, 2])
    ax2.plot(uav_trajectory[:, 2])
    ax2.set_ylabel("z (m)")

    ax3 = fig.add_subplot(414)
    ax3.plot(uav_des_traj[:, 8])
    ax3.plot(uav_trajectory[:, 8])
    ax3.set_ylabel("psi (rad)")

    fig.suptitle(title, fontsize=16)

    plt.show()
