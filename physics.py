import numpy as np
from numpy.typing import NDArray
from typing import List
from enum import Enum

class constants ():
    dt : float = 0.1
    g : float = 9.81

class frames(Enum):
    W = 0
    B = 1

class quaternion() :
    value : NDArray = np.array((4,))
    
    def  __init__ (self, value : NDArray):
        assert(value.shape == (4,))
        norm = np.linalg.norm(value)
        self.value = value

    def hamilton_product(self, q2: 'quaternion') -> 'quaternion':
        """Quaternion multiplication (Hamilton product): q = q1 * q2"""
        w1, x1, y1, z1 = self.value
        w2, x2, y2, z2 = q2.value
        return quaternion(np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ]))
    
    def inverse(self) -> 'quaternion':
        return quaternion(np.hstack([np.array([self.value[0]]), -self.value[1:]]))

    def rotate_vec(self, vector: NDArray) -> NDArray:
        """
        Rotate the given vector around the quaternion using the formula quat(0,-v) = H(H(q, quat(0, v)), q.inv())
        """
        assert vector.shape == (3,) or vector.shape == (3, 1)
        if vector.shape == (3, 1):
            vector = vector.flatten()
        v_quat = quaternion(np.hstack([0, vector]))
        q_inv = self.inverse()
        rotated = self.hamilton_product(v_quat).hamilton_product(q_inv)
        return -rotated.value[1:]
        
class physics_object ():
    # Physical Properties
    W_p_COM : NDArray # [m]
    W_dp_COM : NDArray # [m]
    W_ddp_COM : NDArray # [m]

    W_q_B : quaternion # unit
    W_w_B : NDArray
    W_dw_B : NDArray

    mass : float # [kg]
    diameter : float # [m]
    inertia_COM : NDArray # [kg * m^2]

    def __init__ (self, position : NDArray, orientation: quaternion = quaternion(np.array([1.0, 0.0, 0.0, 0.0])), mass : float = 1, diameter : float = 1e-2):
        assert(mass > 0 and diameter > 0)

        self.W_p_COM = position
        self.W_dp_COM : NDArray = np.zeros(3)
        self.W_ddp_COM : NDArray = np.zeros(3)

        self.W_q_B = orientation
        self.W_w_B = np.zeros(3) 
        self.W_dw_B = np.zeros(3)

        self.mass = mass
        self.diameter = diameter
        self.inertia_COM = np.eye(3) * 2 / 5 * mass * (diameter/2)**2 # [kg * m^2] SPHERICAL MODEL

    def update_quat_from_ang_vel(self):
        assert self.W_w_B.shape == (3,) or self.W_w_B.shape == (3,1), f"dimension of w is incorrect {self.W_w_B.shape}" 
        w_norm = np.linalg.norm(self.W_w_B)
        if w_norm != 0:
            theta = w_norm * constants.dt
            axis = self.W_w_B / w_norm
            dq = quaternion(np.hstack((np.cos(theta / 2), axis * np.sin(theta / 2))))
            self.W_q_B = quaternion.hamilton_product(dq, self.W_q_B)
            self.W_q_B = quaternion(self.W_q_B.value / np.linalg.norm(self.W_q_B.value))

    def physics_step(self, frame: frames = frames.W , forces : NDArray = np.zeros((1,3)), application_points : NDArray = np.zeros((1,3))):
        """
        in-args :
         - forces NDArray shape (n, 3) # in body frame
         - application_points (n, 3) # in body frame
        """
        if (forces[:].shape == (3,1)):
            forces = np.reshape(forces, (3,))

        if (application_points.shape == (3,) or application_points.shape == (3,1)):
            application_points = np.reshape(application_points, (1,3))
        
        n = forces.shape[0]
        assert application_points.shape[0] == n, f"got unequal amounts of forces and points: {n} != {application_points.shape[0]}"
        assert forces.shape[1] == 3 and application_points.shape[1] == 3, f"forces or points are not 3D: {forces.shape[1]} and {application_points.shape[1]}"
        assert len(forces.shape) == 2 and len(application_points.shape) == 2, f"got too many indices: {forces.shape} and {application_points.shape}" 

        # Find net change in momentum at COM
        net_force = np.zeros(3)
        net_torque = np.zeros(3)
        for i in range(n):
            net_force = net_force + forces[i]

        if frame == frames.B:
            print(f"Before tansform: {net_force}, orientation: {self.W_q_B.value}")
            net_force = self.W_q_B.rotate_vec(net_force)
            print(f"After tansform: {net_force}")

            for i in range(n):
                net_torque = net_torque + np.cross(application_points[i], forces[i])

            net_torque = self.W_q_B.rotate_vec(net_force)
        elif frame == frames.W:
            for i in range(n):
                net_torque = net_torque + np.cross(application_points[i] - self.W_p_COM, forces[i])

            
        # Acceleration
        self.W_ddp_COM = net_force / self.mass
        self.W_dw_B = np.linalg.inv(self.inertia_COM) @ (net_torque - np.cross(self.W_w_B, self.inertia_COM @ self.W_w_B)) # Note: Transport Theorem because of rotating Body Frame

        # Velocity
        self.W_dp_COM += self.W_ddp_COM * constants.dt
        self.W_w_B += self.W_dw_B * constants.dt

        # Absolute
        self.W_p_COM += self.W_dp_COM * constants.dt
        self.update_quat_from_ang_vel()







