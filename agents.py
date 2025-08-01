from environment import line_map, node
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from physics import physics_object, constants
import cvxpy as opt
from scipy.linalg import solve_discrete_are, expm
import heapq
import random

class agent_parameters():

    # Solving Parameters
    horizon_length: int = 10
    state_weight: float = 1e1
    input_weight: float = 1e0
    input_actuation_limit: float = 5e0
    minimum_hallway_width: float = 0.1
    goal_score_modifier: float = float('-inf')

    # Default Values
    default_name: str = "Unnamed Agent"
    default_mass: float= 1
    default_diameter: float = 0.0
    default_path_length_weight: float = 1.0
    default_advert_weight: float = 1.0

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
    UNvisited_nodes : List[node]

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
        self.diameter = diameter #unused
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
        self.UNvisited_nodes = []
        for idx in self.current_node.connectivity:
            neighbor: node = self.global_map.nodes[idx]
            self.UNvisited_nodes.append(neighbor)
            self.personal_map.add_node(neighbor)
            self.personal_map.add_connection(self.current_node.node_idx, idx)

        # random.seed(42)
        self.target_node = self.UNvisited_nodes[round(random.random()*(len(self.UNvisited_nodes)-1))]
        # print(f"{self} has starting options {[n.node_idx for n in self.UNvisited_nodes]} and chose {self.target_node}")
        
        # LTI Model
        A_c = np.block([[np.zeros((2,2)),np.eye(2)],[-K/mass,-D/mass]])
        B_c = np.block([[np.zeros((2,2))],[np.eye(2)/mass]])
        M   = expm(np.block([[A_c,B_c], [np.zeros((2,6))]]) * constants.dt)
        M = M.round(decimals=2) # Numerical disc, is not very accurate, likely floating point errors
        self.A = M[:4,:4]
        self.B = M[:4,4:6]
        self.C = np.block([np.eye(2),np.zeros((2,2))])

        # Cost Function Weights
        self.Q = agent_parameters.state_weight * np.eye(4)
        self.R = agent_parameters.input_weight * np.eye(2)
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        
    def __str__ (self):
        return f"Agent \"{self.name}\" at \nx:  {self.W_p_COM[0]}\ny:  {self.W_p_COM[1]}\ndx: {self.W_p_COM[2]}\ndy: {self.W_p_COM[3]}\n"
    
    def advertise(self) -> Tuple[float, Tuple[float, float]]:
        return self.score, (float(self.W_p_COM[0]), float(self.W_p_COM[1]))
    
    def find_best_target_node(self, adverts : List[Tuple[float, Tuple[float, float]]] = [], path_length_weight: float = agent_parameters.default_path_length_weight, advert_weight: float = agent_parameters.default_advert_weight) -> node:
        """
        Graphsearch for the node that has the smallest sum of distance to current node and all agent scores divided by the distnace form this agent to them
        """
        if self.current_node.name == "GOAL Node":
            self.score = agent_parameters.goal_score_modifier
            return self.current_node
        
        if len(self.UNvisited_nodes) == 1:
            return self.UNvisited_nodes[0]

        def absolute_distance(n1: node, n2: node) -> float:
            return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        
        def heuristic(personal_score: float, candidate: node, adverts: List[Tuple[float, Tuple[float, float]]]) -> float:
            result: float = 0.0
            for score, (x, y) in adverts:
                result += score / (personal_score * (1 + np.sqrt((candidate.x - x)**2 + (candidate.y - y)**2)))
                """
                if score < 0:
                    return node closest to that advert, also possible
                """
            return result
        
        def length_of_path(path: List[int]) -> float:
            if len(path) <= 1:
                return 0.0
            result: float = 0.0
            start_id: int = path.pop(0)
            for end_id in path:
                result += absolute_distance(self.global_map.nodes[start_id], self.global_map.nodes[end_id])
                start_id = end_id
            return result

        candidates = []

        if path_length_weight == 0.0: # save computations for default case
            for candidate in self.UNvisited_nodes:
                score: float = advert_weight * heuristic(self.score, candidate, adverts)
                heapq.heappush(candidates, (score, random.random(), candidate)) # Random is a tie braker to split up agents that go to the same direction   
        else:
            for candidate in self.UNvisited_nodes:
                path: List[int] = self.personal_map.astar(self.current_node.node_idx, candidate.node_idx)
                score: float = path_length_weight * length_of_path(path) + advert_weight * heuristic(self.score, candidate, adverts)
                heapq.heappush(candidates, (score, random.random(), candidate)) # Random is a tie braker to split up agents that go to the same direction   

        _ , _ , best_candidate = heapq.heappop(candidates)
        # assert best_candidate is not None, f"found None candidate when looking in {self.UNvisited_nodes}"
        return best_candidate

    def update(self, force : NDArray = np.zeros((2,1)), adverts: List[Tuple[float, Tuple[float, float]]]= []) -> None:
        """
        Main function for progressing an agent through time.
        Handles all graph processes as well as applying the force specified.
        """
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
                    self.target_node = self.find_best_target_node(adverts= adverts)
                print(f"Agent {self.name} has reached the target node {self.current_node}, now targetting {self.target_node}")

    def find_first_intermediate_target(self) -> node:
        path = self.personal_map.astar(self.current_node.node_idx, self.target_node.node_idx)
        if len(path) == 1:
            return self.global_map.nodes[path[0]]
        return self.global_map.nodes[path[1]]
    
    def find_input(self, reference: NDArray = np.zeros((2,))) -> NDArray:
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
    
    """ CVXPY DCP doesnt allow for convex >= affine, this aspect will be ignored
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
    """