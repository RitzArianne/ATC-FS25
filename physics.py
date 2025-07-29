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

class physics_object ():
    # Physical Properties
    W_p_COM : NDArray # [m]

    mass : float # [kg]
    diameter : float # [m]

    # LTI Model
    A : NDArray
    B : NDArray
    C : NDArray

    # Cost Function Weights
    P : NDArray
    Q : NDArray
    R : NDArray

    # Feedback Law
    F : NDArray

    def __init__ (self, position : NDArray, mass : float = 1, diameter : float = 0, K: NDArray = np.eye(2), D: NDArray = np.zeros(2)):
        assert(mass > 0 and diameter >= 0)
        self.W_p_COM = np.block(position.reshape((2,1)), np.zeros(2,1))

        self.mass = mass
        self.diameter = diameter
        # LTI Model
        self.A = np.block([[np.zeros((2,2)),np.eye(2)],[K/mass,D/mass]])
        self.B = np.block([[np.zeros((2,2))],[np.eye(2)/mass]])
        self.C = np.block([np.eye(2),np.zeros((2,2))])

    def physics_step(self, force : NDArray = np.zeros((2,1))):
        """
        in-args :
         - force NDArray shape (2,1) # world frame
        """
        assert force.size == 2, f"invalid for {force}"
        force = force.reshape((2,1))
        self.W_p_COM = self.A @ self.W_p_COM + self.B @ force
        assert self.W_p_COM.shape == (4,1), f"{self.W_p_COM}"