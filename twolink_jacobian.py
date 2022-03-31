import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

### -----------------------------------------------------------------------------


# Jacobian Matrix
def Jacob(th, l):
    l1 = l[0]
    l2 = l[1]
    th1 = th[0]
    th2 = th[1]
    J = np.array(
        [
            [-l1 * np.sin(th1) - l2 * np.sin(th1 + th2), -l2 * np.sin(th1 + th2)],
            [l1 * np.cos(th1) + l2 * np.cos(th1 + th2), l2 * np.cos(th1 + th2)],
        ]
    )
    return J


# forward kinematics
def forw_kin(th, l):
    l1 = l[0]
    l2 = l[1]
    th1 = th[0]
    pos = np.array(
        [
            l2 * np.cos(th.sum()) + l1 * np.cos(th1),
            l2 * np.sin(th.sum()) + l1 * np.sin(th1),
        ]
    )
    return pos


# inverse kinematics using
def sim(th, dt):
    k = 0.5
    finish = 0.001
    y = th
    while True:
        # forward kinematics
        pfwd = forw_kin(th, l)

        # Jacobian
        J = Jacob(th, l)

        # dError
        error = Dp - pfwd
        derror = k * (error)

        # Calculate dth
        omg = np.matmul(np.linalg.pinv(J), derror)

        # integral dth
        th = th + dt * omg

        # plog
        y = np.vstack((y, th))

        # termination
        if np.linalg.norm(error) <= finish:
            break
    return y


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

    # target point
    pr = 1
    p_th = np.pi
    Dp = np.array([pr * np.cos(p_th), pr * np.sin(p_th)])  # target point x,y coordinate

    # State
    L1 = 1.0  # length of arm 1 in m
    L2 = 1.0  # length of arm 2 in m
    l = np.array([L1, L2])
    l_th = np.array([np.pi / 4, np.pi / 4])
    ini_state = l_th

    # numerical simulation parameter
    dt = 0.1

    ### inverse kinematics
    y = sim(ini_state, dt)

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
