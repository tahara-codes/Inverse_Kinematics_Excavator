import numpy as np
import sympy as sm
from sympy.physics.mechanics import dynamicsymbols
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from control_joy import JoyController


class ExcavatorInverseKinematics:
    def __init__(self):

        # link parameters
        L2 = 6.866
        L3 = 4.226
        L4 = 1.7
        self.l = np.array([L2, L3, L4])

        # initial joint state
        th2 = np.deg2rad(45.0)  # 45
        th3 = np.deg2rad(-135.0)  # 90
        th4 = np.deg2rad(0.0)
        self.th = np.array([th2, th3, th4])
        print("initial th: ", self.th)

        # target point x,y coordinate
        x, z = self.forward_kinematics(self.l, self.th)
        self.target = np.array([x, z])

        # numerical simulation parameters
        self.dt = 0.2
        self.k = 0.5
        self.finish = 0.20

        # define the sysmbols
        self.l2, self.l3, self.l4 = dynamicsymbols("l2 l3 l4")
        self.theta2, self.theta3, self.theta4 = dynamicsymbols("theta2 theta3 theta4")

        # forward kinematics
        self.px, self.pz = self.forward_kinematics(
            [self.l2, self.l3, self.l4],
            [self.theta2, self.theta3, self.theta4],
        )

        # differentiation of Jacobian matrix
        self.Jacobian()

        # Controller
        self.joy = JoyController()
        self.scale = 0.2

        # graph
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)

    def main(self):
        try:
            while True:
                # change target position
                self.joy.get_controller_value()
                dx = -self.joy.l_hand_x * self.scale
                dz = self.joy.l_hand_y * self.scale
                self.target = self.target + [dx, dz]

                # solve inverse kinematics
                self.simulation(self.target, self.th)

        except KeyboardInterrupt:
            exit()

    def simulation(self, target, theta):
        l = self.l
        th = theta

        while True:
            x = self.subs(self.px, l, th)
            z = self.subs(self.pz, l, th)
            p = sm.Matrix([x, z])
            pfwd = np.array(p).astype(np.float64).flatten()
            print("pfwd: ", pfwd)

            # Jacobian
            J = self.subs(self.Jspl, l, th)
            J = np.array(J).astype(np.float64)
            # Jp = np.matmul(
            #     np.transpose(J), np.linalg.pinv(np.matmul(J, np.transpose(J)))
            # )
            Jp = np.linalg.pinv(J)
            # print("Jp: ", Jp)

            # dError
            error = target - pfwd
            derror = self.k * (error)

            # Calculate J
            omg = np.matmul(Jp, derror)

            # integral dth
            th = th + self.dt * omg

            # realtime plot
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            self.ax.set_zlim(-10, 10)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.scatter(pfwd[0], 0.0, pfwd[1])
            self.ax.plot([0, pfwd[0]], [0, 0], [0, pfwd[1]], "-")
            plt.draw()
            plt.pause(0.001)
            plt.cla()

            # termination
            if np.linalg.norm(error.astype(np.float64)) <= self.finish:
                self.th = th
                break

    def subs(self, p, l, th):
        return p.subs(
            zip(
                [
                    self.l2,
                    self.l3,
                    self.l4,
                    self.theta2,
                    self.theta3,
                    self.theta4,
                ],
                [l[0], l[1], l[2], th[0], th[1], th[2]],
            )
        )

    def forward_kinematics(self, l, theta):
        l2 = l[0]
        l3 = l[1]
        l4 = l[2]
        theta2 = theta[0]
        theta3 = theta[1]
        theta4 = theta[2]

        px = (
            l4 * sm.cos(theta2 + theta3 + theta4)
            + l3 * sm.cos(theta2 + theta3)
            + l2 * sm.cos(theta2)
        )
        pz = (
            l4 * sm.sin(theta2 + theta3 + theta4)
            + l3 * sm.sin(theta2 + theta3)
            + l2 * sm.sin(theta2)
        )

        return px, pz

    def Jacobian(self):
        a11 = sm.diff(self.px, self.theta2)
        a12 = sm.diff(self.px, self.theta3)
        a13 = sm.diff(self.px, self.theta4)
        a21 = sm.diff(self.pz, self.theta2)
        a22 = sm.diff(self.pz, self.theta3)
        a23 = sm.diff(self.pz, self.theta4)

        # Jabobian matrix
        J = sm.Matrix([[a11, a12, a13], [a21, a22, a23]])
        self.Jspl = sm.simplify(J)


if __name__ == "__main__":
    excavator_inverse_kinematics = ExcavatorInverseKinematics()
    excavator_inverse_kinematics.main()
