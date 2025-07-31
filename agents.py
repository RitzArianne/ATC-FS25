from environment import line_map, node
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from physics import physics_object, constants
import cvxpy as opt
from scipy.linalg import solve_discrete_are, expm

# from random 
import random

class agent_parameters():

    # Solving Parameters
    horizon_length: int = 10
    state_weight: float = 1e1
    input_weight: float = 1e0
    input_actuation_limit: float = 5e0
    minimum_hallway_width: float = 0.1

    # Default Values
    default_name: str = "Unnamed Agent"
    default_mass: float= 1
    default_diameter: float = 0.0

class agent(physics_object) :

    # General
    name : str
    max_calc_steps : int
    score : float

    # Vision
    global_map : line_map
    personal_map : line_map

    # Positioning
    current_node : node
    target_node : node 
    UNvisited_nodes : List[node] = []

    def __init__ (
            self, 
            global_map: line_map, 
            position : NDArray, 
            mass : float = agent_parameters.default_mass, 
            diameter : float = agent_parameters.default_diameter, 
            K: NDArray = np.zeros((2,2)), 
            D: NDArray = np.eye(2)
            ):
        assert(mass > 0 and diameter >= 0)

        # General
        self.mass = mass
        self.diameter = diameter
        self.name = agent_parameters.default_name
        self.max_calc_steps = agent_parameters.horizon_length
        self.score = 0.0

        # Vision
        self.personal_map = line_map()
        self.global_map = global_map

        # Positioning
        self.W_p_COM = np.vstack([position.reshape((2,1)), np.zeros((2,1))])
        self.current_node, _ = global_map.find_closest_node(self.W_p_COM[0], self.W_p_COM[1])
        self.personal_map.add_node(self.current_node)
        for idx in self.current_node.connectivity:
            neighbor: node = self.global_map.nodes[idx]
            self.UNvisited_nodes.append(neighbor)
            self.personal_map.add_node(neighbor)
            self.personal_map.add_connection(self.current_node.node_idx, idx)

        random.seed(42)
        self.target_node = self.UNvisited_nodes[round(random.random()*(len(self.UNvisited_nodes)-1))]
        
        # LTI Model
        A_c = np.block([[np.zeros((2,2)),np.eye(2)],[-K/mass,-D/mass]])
        B_c = np.block([[np.zeros((2,2))],[np.eye(2)/mass]])
        M   = expm(np.block([[A_c,B_c], [np.zeros((2,6))]]) * constants.dt)
        M = M.round(decimals=2) # Numerical disc, is not very accurate 
        self.A = M[:4,:4]
        self.B = M[:4,4:6]
        self.C = np.block([np.eye(2),np.zeros((2,2))])

        # Cost Function Weights
        self.Q = agent_parameters.state_weight * np.eye(4)
        self.R = agent_parameters.input_weight * np.eye(2)
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        
    def __str__ (self):
        return f"Agent \"{self.name}\" at \nx:  {self.W_p_COM[0]}\ny:  {self.W_p_COM[1]}\ndx: {self.W_p_COM[2]}\ndy: {self.W_p_COM[3]}\n"
        #return f"{self.W_p_COM}"
    
    def advertise(self) -> Tuple[NDArray, float]:
        """
        Give away agent infromation:
        Positon:
        Score:
        """
        return self.W_p_COM, self.score
    
    def find_best_target_node(self, adverts : NDArray = np.zeros((0,0))) -> node:
        """
        Graphsearch for the node that has the smallest sum of distance to current node and all agent scores divided by the distnace form this agent to them
        """
        if self.current_node.name == "GOAL Node":
            return self.current_node
        # TODO: actually implement this. Might need to be outside of update tho for sync reasons
        return self.UNvisited_nodes[0]

    def update(self, force : NDArray = np.zeros((2,1))) -> None:
        self.physics_step(force=force)
        assert self.W_p_COM.shape == (4,1), f"{self.W_p_COM}"

        last_node = self.current_node
        closest_node, distance = self.global_map.find_closest_node(self.W_p_COM[0], self.W_p_COM[1])

        if distance < agent_parameters.minimum_hallway_width and closest_node != last_node:
            self.current_node = closest_node
            print(f"agent {self.name} has reached a new node {last_node} -> {self.current_node}")
            if self.current_node in self.UNvisited_nodes:
                self.UNvisited_nodes.remove(self.current_node)
                for idx in self.current_node.connectivity:
                    neighbor = self.global_map.nodes[idx]
                    if neighbor not in self.personal_map.nodes:
                        self.UNvisited_nodes.insert(0,neighbor)
                        self.personal_map.add_node(neighbor)
                        self.personal_map.add_connection(self.current_node.node_idx, neighbor.node_idx)
            elif self.current_node in self.personal_map.nodes:
                pass
            else:
                raise Exception(f"Node without connectivity found: {last_node} -> {self.current_node}\n{self.personal_map.print_map()}")
            
            if self.current_node == self.target_node:
                if len(self.UNvisited_nodes) > 0:
                    self.target_node = self.find_best_target_node()
                print(f"Agent {self.name} has reached the target node {self.current_node}, now targetting {self.target_node}")

    def find_first_intermediate_target(self) -> node:
        path = self.personal_map.astar(self.current_node.node_idx, self.target_node.node_idx)
        if len(path) == 1:
            return self.global_map.nodes[path[0]]
        # assert next_node_idx in self.current_node.connectivity, f"Found intermediate Target with idx {next_node_idx}, for current idx {self.current_node.node_idx} with connectivity {self.current_node.connectivity}"
        return self.global_map.nodes[path[1]] # Can cause error if no path is found (empty list returned)
    
    def find_input(self, reference: NDArray = np.zeros((2,)), print_solver: bool = False) -> NDArray:
        N = self.max_calc_steps
        x = opt.Variable((N+1,4), name= "state")
        u = opt.Variable((N,2), name = "input")
        r = np.vstack((reference.reshape((2,1)), np.zeros((2,1)))).reshape((4,))

        objective = opt.Minimize(
            opt.norm2(self.P @ (x[N] - r)) + 
            sum(opt.norm2(self.Q**0.5 @ (x[i] - r)) for i in range(N)) + 
            sum(opt.norm2(self.R**0.5 @ u[i]) for i in range(N))
        )

        constraints = []
        constraints.append(x[0] == self.W_p_COM.reshape((4,)))
        constraints.extend([x[i+1] == self.A @ x[i] + self.B @ u[i] for i in range(N)])
        constraints.extend([opt.norm2(x[i,2:4]) <= agent_parameters.minimum_hallway_width * agent_parameters.horizon_length for i in range (N)])
        constraints.extend([opt.norm2(u[i]) <= agent_parameters.input_actuation_limit for i in range (N)])

        # Constraint to be certain distance from current line segment

        for i in range(N+1):
            pos = x[i, :2]  # agent position
            t = opt.Variable(name= f"Line variable")
            x1, y1, x2, y2 = self.current_node.x, self.current_node.y, reference[0], reference[1]
            xt = (1 - t) * x1 + t * x2
            yt = (1 - t) * y1 + t * y2
            proj = opt.hstack([xt, yt])
            constraints += [opt.norm(pos - proj, p = 2) <= agent_parameters.minimum_hallway_width, t >= 0, t <= 1]

        prob = opt.Problem(objective, constraints)
        rresult = prob.solve(solver='SCS')
        self.score = prob.value

        # print(f"Found input is {u[0].value}, which should lead to {x[N].value} in the future\n")
        # print(f"Expected Trajectory:\n{x[:].value}")
        return np.array(u[0].value)
    
    def find_maximum_travel_distance(self) -> float:
        N = self.max_calc_steps
        x = opt.Variable((N+1,4), name= "state")
        u = opt.Variable((N,2), name = "input")

        objective = opt.Maximize(opt.norm2(x[N,:2]-x[0,:2]))

        constraints = []
        constraints.append(x[0] == np.zeros(4))
        constraints.extend([x[i+1] == self.A @ x[i] + self.B @ u[i] for i in range(N)])
        constraints.extend([opt.norm2(x[i,2:4]) <= agent_parameters.minimum_hallway_width * agent_parameters.horizon_length for i in range (N)])
        constraints.extend([opt.norm2(u[i]) <= agent_parameters.input_actuation_limit for i in range (N)])

        prob = opt.Problem(objective, constraints)
        result = prob.solve(solver='SCS')

        return prob.value
