from environment import line_map, node
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from physics import physics_object, frames, constants
import cvxpy as opt
from scipy.linalg import solve_discrete_are, expm

class agent_parameters():
    horizon_length: int = 10
    state_weight: float = 1e1
    input_weight: float = 1e0
    input_actuation_limit: float = 1e1

class agent(physics_object) :
    name : str
    global_map : line_map
    personal_map : line_map

    closest_node : node
    target_node : node 

    visited_nodes : List[node] = []
    UNvisited_nodes : List[node] = []

    max_calc_steps : int
    score : float

    def __init__ (self, global_map: line_map, position : NDArray, mass : float = 1, diameter : float = 0, K: NDArray = np.eye(2), D: NDArray = np.zeros((2,2))):
        assert(mass > 0 and diameter >= 0)

        self.W_p_COM = np.vstack([position.reshape((2,1)), np.zeros((2,1))])

        self.mass = mass
        self.diameter = diameter
        # LTI Model
        A_c = np.block([[np.zeros((2,2)),np.eye(2)],[K/mass,D/mass]])
        B_c = np.block([[np.zeros((2,2))],[np.eye(2)/mass]])
        M   = expm(np.block([[A_c,B_c], [np.zeros((2,6))]]) * constants.dt)

        self.A = M[:4,:4]
        self.B = M[:4,4:6]
        self.C = np.block([np.eye(2),np.zeros((2,2))])
        # Cost Function Weights
        self.Q = agent_parameters.state_weight * np.eye(4)
        self.R = agent_parameters.input_weight * np.eye(2)
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        
        self.name = "NoName"
        self.global_map = global_map
        self.personal_map = line_map()
        self.closest_node, _ = global_map.find_closest_node(self.W_p_COM[0], self.W_p_COM[1])
        self.personal_map.add_node(self.closest_node)

        self.UNvisited_nodes.append(self.global_map.nodes(idx) for idx in self.closest_node.connectivity)

        self.max_calc_steps = agent_parameters.horizon_length
        self.score = 0

    def __str__ (self):
        return f"Agent \"{self.name}\" at \nx: {self.W_p_COM[0]}\ny: {self.W_p_COM[1]}\ndx: {self.W_p_COM[2]}\ndy: {self.W_p_COM[3]}\n"
        #return f"{self.W_p_COM}"
    
    def advertise(self) -> Tuple[NDArray, float]:
        """
        Give away agent infromation:
        Positon:
        Score:
        """
        return self.W_p_COM, self.score

    def update(self, force : NDArray = np.zeros((2,1))):
        self.physics_step(force=force)

    def find_input(self, reference: NDArray = np.zeros((2,)), verbose: bool = False) -> NDArray:
        N = self.max_calc_steps
        x = opt.Variable((N+1,4))
        u = opt.Variable((N,2))
        r = np.vstack((reference.reshape((2,1)), np.zeros((2,1)))).reshape((4,))

        objective = opt.Minimize(
            opt.norm2(self.P @ (x[N] - r)) + 
            sum(opt.norm2(self.Q**0.5 @ (x[i] - r)) for i in range(N)) + 
            sum(opt.norm2(self.R**0.5 @ u[i]) for i in range(N))
        )

        constraints = []
        constraints.append(x[0] == self.W_p_COM.reshape((4,)))
        constraints.extend([x[i+1] == self.A @ x[i] + self.B @ u[i] for i in range(N)])
        constraints.extend([opt.norm2(u[i]) <= agent_parameters.input_actuation_limit for i in range (N)])

        problem = opt.Problem(objective, constraints)
        
        result = problem.solve(verbose=verbose)
        self.score = problem.value

        print(f"Found input is {u[0].value}, which should lead to {x[N].value} in the future\n")
        print(f"Expected Trajectory:\n{x[:].value}")
        return np.array(u[0].value)
