from environment import line_map, node
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from physics import physics_object, quaternion, frames

class agent(physics_object) :
    name : str
    global_map : line_map
    personal_map : line_map

    closest_node : node
    target_node : node 

    visited_nodes : List[node] = []

    def __init__ (self, global_map: line_map, position : NDArray, orientation: quaternion = quaternion(np.array([1.0, 0.0, 0.0, 0.0])), mass : float = 1, diameter : float = 1e-2):
        assert(mass > 0 and diameter > 0, f"illegal values {mass, diameter}")

        self.W_p_COM = position
        self.W_dp_COM : NDArray = np.zeros(3)
        self.W_ddp_COM : NDArray = np.zeros(3)

        self.W_q_B = orientation
        self.W_w_B = np.zeros(3) 
        self.W_dw_B = np.zeros(3)

        self.mass = mass
        self.diameter = diameter
        self.inertia_COM = np.eye(3) * 2 / 5 * mass * (diameter/2)**2 # [kg * m^2] SPHERICAL MODEL
        
        self.name = "NoName"
        self.global_map = global_map
        self.personal_map = line_map()
        self.closest_node, _ = global_map.find_closest_node(self.W_p_COM[0], self.W_p_COM[1])
        self.personal_map.add_node(self.closest_node)

    def __str__ (self):
        return f"Agent \"{self.name}\" at \nx: {self.W_p_COM[0]}\ny: {self.W_p_COM[1]}\nz: {self.W_p_COM[2]}\norientation: {self.W_q_B.value}\n"

    def update(self):
        self.physics_step(forces=np.array([[1, 0, 0]]), frame= frames.B)





