import numpy as np
import sympy as sm
from sympy.physics.mechanics import dynamicsymbols
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D


class ExcavatorIK:
    def __init__(self):

        # link parameters
        L1 = 0.0
        L2 = 6.866
        L3 = 4.226
        self.l_list = np.array([L1, L2, L3])

        # kinematics by sympy
        sym_th1, sym_th2, sym_th3 = dynamicsymbols("theta1 theta2 theta3")
        self.sym_theta_list = [sym_th1, sym_th2, sym_th3]
        px, py, pz = self.forward_kinematics(self.l_list, self.sym_theta_list)
        self.sym_p_list = [px, py, pz]
        self.J = self.Jacobian(self.sym_p_list, self.sym_theta_list)

        # numerical simulation parameters
        self.dt = 0.5
        self.finish = 0.01

        # graph
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)

        self.th = []

    def action(self, target):
        x = self.subs(self.sym_p_list[0], self.theta)
        y = self.subs(self.sym_p_list[1], self.theta)
        z = self.subs(self.sym_p_list[2], self.theta)
        pfwd = sm.Matrix([x, y, z])
        pfwd = np.array(pfwd).astype(np.float64).flatten()
        print("target current pos: ", target, pfwd)

        # Jacobian
        J = self.subs(self.J, self.theta)
        J = np.array(J).astype(np.float64)
        Jinv = np.linalg.inv(J)

        # Calculate J
        error = self.dt * (target - pfwd)
        omg = np.matmul(Jinv, error)

        # integral dth
        self.theta = self.theta + self.dt * omg
        print("Current th: ", np.array(self.theta))

        # Graph
        self.Graph3D(pfwd)

        # termination
        done = False
        if np.linalg.norm(error.astype(np.float64)) <= self.finish:
            done = True

        action = omg
        return action, done

    def subs(self, p, th):
        return p.subs(
            zip(
                self.sym_theta_list,
                [th[0], th[1], th[2]],
            )
        )

    def forward_kinematics(self, l_list, th_list):
        l1 = l_list[0]
        l2 = l_list[1]
        l3 = l_list[2]
        th1 = th_list[0]
        th2 = th_list[1]
        th3 = th_list[2]
        px = sm.cos(th1) * (+l3 * sm.cos(th2 + th3) + l2 * sm.cos(th2) + l1)
        py = sm.sin(th1) * (+l3 * sm.cos(th2 + th3) + l2 * sm.cos(th2) + l1)
        pz = +l3 * sm.sin(th2 + th3) + l2 * sm.sin(th2)
        return px, py, pz

    def Jacobian(self, p, th):
        a11 = sm.diff(p[0], th[0])
        a12 = sm.diff(p[0], th[1])
        a13 = sm.diff(p[0], th[2])
        a21 = sm.diff(p[1], th[0])
        a22 = sm.diff(p[1], th[1])
        a23 = sm.diff(p[1], th[2])
        a31 = sm.diff(p[2], th[0])
        a32 = sm.diff(p[2], th[1])
        a33 = sm.diff(p[2], th[2])
        J = sm.Matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        return sm.simplify(J)

    def Graph3D(self, pfwd):
        # realtime plot
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-10, 10)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.scatter(pfwd[0], pfwd[1], pfwd[2])
        self.ax.plot([0, pfwd[0]], [0, pfwd[1]], [0, pfwd[2]], "-")
        plt.draw()
        plt.pause(0.001)
        plt.cla()

    def main(self):
        target = [4.0, 0.0, 3.0]
        th1 = np.deg2rad(0.0)  # 45
        th2 = np.deg2rad(45.0)  # 45
        th3 = np.deg2rad(-135.0)  # 90
        self.theta = np.array([th1, th2, th3])

        while True:
            action, done = self.action(target)
            if done is True:
                break


if __name__ == "__main__":
    excavator_ik = ExcavatorIK()
    excavator_ik.main()
