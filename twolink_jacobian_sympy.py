import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sm
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.quantum import TensorProduct


### -----------------------------------------------------------------------------
def simulation(th, dt):
    # simulation parameters
    k = 0.5
    finish = 0.1

    # plot
    data = th

    # define the sysmbols
    l1, l2 = dynamicsymbols("l1 l2")
    theta1, theta2 = dynamicsymbols("theta1 theta2")

    # forward kinematics
    px = l1 * sm.cos(theta1) + l2 * sm.cos(theta1 + theta2)
    py = l1 * sm.sin(theta1) + l2 * sm.sin(theta1 + theta2)

    # differentiation of Jacobian matrix
    a11 = sm.diff(px, theta1)
    a12 = sm.diff(px, theta2)
    a21 = sm.diff(py, theta1)
    a22 = sm.diff(py, theta2)

    # Jabocian matrix
    J = sm.Matrix([[a11, a12], [a21, a22]])
    Jspl = sm.simplify(J)

    while True:
        # forward kinematics
        x = px.subs(zip([l1, l2, theta1, theta2], [l[0], l[1], th[0], th[1]]))
        y = py.subs(zip([l1, l2, theta1, theta2], [l[0], l[1], th[0], th[1]]))
        p = sm.Matrix([x, y])
        pfwd = np.array(p).astype(np.float64).flatten()

        # Jacobian
        J = Jspl.subs(zip([l1, l2, theta1, theta2], [l[0], l[1], th[0], th[1]]))
        J = np.array(J).astype(np.float64)

        # dError
        error = Dp - pfwd
        derror = k * (error)

        # Calculate dth
        omg = np.matmul(np.linalg.inv(J), derror)

        # integral dth
        th = th + dt * omg

        # plog
        data = np.vstack((data, th))

        # termination
        if np.linalg.norm(error) <= finish:
            break

    return data


### -----------------------------------------------------------------------------
# init data
def init():
    line.set_data([], [])
    linepath.set_data([], [])
    line1path.set_data([], [])
    time_text.set_text("")
    return line, linepath, line1path, time_text


# graph animation
def animate(i):
    thisx = [0, x1[i], x2[i]]  # x codrin of point making arm
    thisy = [0, y1[i], y2[i]]  # y cord of point
    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    linepath.set_data(x2[:i], y2[:i])
    line1path.set_data(x1[:i], y1[:i])
    return line, linepath, line1path, time_text


### -----------------------------------------------------------------------------


if __name__ == "__main__":

    # target point x,y coordinate
    pr = 1
    p_th = np.pi
    Dp = np.array([pr * np.cos(p_th), pr * np.sin(p_th)])

    # State
    L1 = 1.0
    L2 = 1.0
    l = np.array([L1, L2])
    th = np.array([np.pi / 4, np.pi / 4])
    ini_state = th

    # numerical simulation parameter
    dt = 0.1

    # solve inverse kinematics
    y = simulation(ini_state, dt)

    ### graph
    x1 = L1 * np.cos(y[:, 0])
    y1 = L1 * np.sin(y[:, 0])
    x2 = L2 * np.cos(y[:, 0] + y[:, 1]) + x1
    y2 = L2 * np.sin(y[:, 0] + y[:, 1]) + y1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()

    time_template = "time = %.1fs"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
    (line,) = ax.plot([], [], "o-", lw=2)
    (linepath,) = ax.plot([], [])
    (line1path,) = ax.plot([], [])

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, len(y)), interval=30, blit=True, init_func=init
    )
    ax.plot(Dp[0], Dp[1], marker="o", markersize=3, color="red")
    plt.show()
