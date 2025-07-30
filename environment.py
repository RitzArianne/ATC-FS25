from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from numpy.typing import NDArray

class node () :
    x : float = 0
    y : float = 0
    name : str = "Unnamed Node"
    node_idx : int
    connectivity : List[int] = []

    def __init__(self, x:float, y:float) :
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.name}(idx: {self.node_idx})"

    def change_name(self, new_name : str) :
        self.name = new_name

    def to_numpy(self) -> NDArray:
        return np.array([self.x, self.y])

class line_map () :
    nodes : list[node]
    connections : List[Tuple[int, int]]
    bounds : Tuple[Tuple[int, int], Tuple[int, int]]

    def __init__(self):
        self.nodes = []
        self.connections = []

    def add_node(self, new_node : node) :
        # TODO: add minimum di
        # stance requirement
        self.nodes.append(new_node)

    def add_connection(self, node_1 : int, node_2 : int) :
        if (len(self.nodes) > max(node_1, node_2) and node_1 >= 0 and node_2 >= 0):
            self.connections.append((node_1, node_2))
        else:
            print(f"Problem adding [{node_1}, {node_2}] to connections when highest idx node is {len(self.nodes) - 1}")

    """ # was found to be unecessary, kept temporarily
    def give_all_connections(self, node_idx) -> List[node]:
        found_nodes = []
        for connection in self.connections:
            if node_idx in connection:
                node_1, node_2 = connection
                found_nodes.append(self.nodes[node_1 if node_1 != node_idx else node_2])
        return found_nodes
    """

    def print_map(self):
        plt.figure()
        print("Plotting all nodes and connections:")
        # Plot all nodes
        xs = [node.x for node in self.nodes]
        ys = [node.y for node in self.nodes]
        plt.scatter(xs, ys, c='red', label='Nodes')

        # Plot connections
        for connection in self.connections:
            node_1 = self.nodes[connection[0]]
            node_2 = self.nodes[connection[1]]
            plt.plot([node_1.x, node_2.x], [node_1.y, node_2.y], c='blue')
    
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Map with Nodes and Connections')
        # plt.legend()
        plt.grid(True)
        plt.show()

    def find_closest_node (self, pos_x : float, pos_y : float) -> Tuple[node, float]:
        # TODO: Needs revision that is not O(n)
        best_distance = float('inf')
        best_node : node
        for candidate in self.nodes:
            distance : float = np.sqrt((pos_x - candidate.x)**2 + (pos_y - candidate.y)**2)
            if distance < best_distance:
                best_distance = distance
                best_node = candidate

        assert best_node, "Somehow no nodes were found in map??"
        return (best_node, best_distance)
    
    def update_node_number_and_connections (self):
        for idx, element in enumerate(self.nodes):
            element.node_idx = idx
            element.connectivity = []
        
        for connection in self.connections:
            self.nodes[connection[0]].connectivity.append(connection[1])
            self.nodes[connection[1]].connectivity.append(connection[0])

    def print_map_and_agents(self, agent_list : List[NDArray] = []):
        plt.figure()
        # Plot all nodes
        xs = [node.x for node in self.nodes]
        ys = [node.y for node in self.nodes]
        plt.scatter(xs, ys, c='red', label='Nodes')

        # Plot connections
        for connection in self.connections:
            node_1 = self.nodes[connection[0]]
            node_2 = self.nodes[connection[1]]
            plt.plot([node_1.x, node_2.x], [node_1.y, node_2.y], c='blue')

        for idx, pos_i in enumerate(agent_list):
            plt.plot(pos_i[0], pos_i[0], "H", c = 'green', label=f"Agent: {idx}")
    
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Map with Nodes and Connections and Agents')
        plt.legend()
        plt.grid(True)
        plt.show()
        


class loss_map(line_map):
    def __init__ (self, scale : float = 1.0):
        super().__init__()
        loss_node_loc = [[0,0],[0,1],[0,-1],[1.0/3,0],[1.0/3,1],[2.0/3,0],[2.0/3,1],[1,0],[1.0/3,-0.5],[1.0/3,-1],[1,-0.5],[-1.0/3,0],[-0.5,0],[-2.0/3,0],[-1,0],[-1.0/3,-1],[-0.5,1],[-2.0/3,-1]]
        loss_node_con = [[0,1],[0,2],[0,3],[3,4],[3,5],[3,8],[5,6],[5,7],[8,10],[8,9],[0,11],[11,15],[11,12],[12,16],[12,13],[13,14],[13,17]]

        for coords in loss_node_loc:
            self.add_node(node(coords[0]*scale, coords[1]*scale))

        for line in loss_node_con:
            self.add_connection(line[0], line[1])

        self.bounds = ((scale * 1,scale * 1), (scale *-1, scale * -1))

        self.update_node_number_and_connections()

class maze_map(line_map):
    def __init__(self, scale: float = 1.0):
        super().__init__()

        maze_node_locs = [
            [0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1],
            [0, 2], [1, 2], [2, 2], [3, 2], [3, 1], [3, 0], [3, -1],
            [2, -1], [1, -1], [0, -1], [0, -2], [1, -2], [2, -2], [2, -3],
            [1, -3], [3, -2]
        ]

        maze_node_conns = [
            [0,1],[1,2],[2,3],[3,4],[4,5],[5,0],
            [5,6],[6,7],[7,8],[8,9],
            [8,10],[10,11],[11,12],[12,13],[13,14],[14,1],
            [14,15],[15,16],
            [14,17],[17,18],[18,19],
            [18,21],[21,12],
            [20,17],
        ]

        for coords in maze_node_locs:
            self.add_node(node(coords[0]*scale, coords[1]*scale))

        for conn in maze_node_conns:
            self.add_connection(conn[0], conn[1])

        self.bounds = ((scale * 3.5, scale * 3.5), (-scale * 3.5, -scale * 3.5))
        self.update_node_number_and_connections()

class complex_maze_map(line_map):
    def __init__(self, scale: float = 1.0):
        super().__init__()

        # 50 Node positions
        maze_node_locs = [
            [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],                 # 0-4
            [4, 1], [3, 1], [2, 1], [1, 1], [0, 1],                 # 5-9 (upper path)
            [0, 2], [1, 2], [2, 2], [3, 2], [4, 2],                 # 10-14
            [2, 3], [3, 3], [4, 3], [1, 3], [0, 3],                 # 15-19
            [-1, 0], [-2, 0], [-3, 0], [-3, 1], [-2, 1],            # 20-24 (left loop)
            [-1, 1], [-1, 2], [-2, 2], [-3, 2], [-3, 3],            # 25-29
            [-2, 3], [-1, 3], [0, 4], [1, 4], [2, 4],               # 30-34
            [3, 4], [4, 4], [4, 5], [3, 5], [2, 5],                 # 35-39
            [1, 5], [0, 5], [-1, 5], [-2, 5], [-3, 5],              # 40-44 (bottom left dead ends)
            [-3, 4], [-2, 4], [-1, 4], [0, 6],                      # 45-48
            [1, 6]                                                  # 49 (top-right exit)
        ]

        # Connections between nodes
        maze_node_conns = [
            [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,0],     # Outer loop
            [9,10],[10,11],[11,12],[12,13],[13,14],[14,5],                  # Top U-turn
            [12,15],[15,16],[16,17],[17,14],                                # Inner top loop
            [11,18],[18,19],[19,10],                                       # Left inner branch
            [0,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,20],        # Left-side loop
            [25,26],[26,27],[27,28],[28,29],[29,30],[30,31],[31,25],       # Left nested loop
            [19,32],[32,33],[33,34],[34,35],[35,36],[36,37],               # Right extension
            [37,38],[38,39],[39,40],[40,41],[41,42],[42,43],[43,44],       # Bottom dead ends
            [44,45],[45,46],[46,47],[47,48],[48,32],                       # Bottom loop reconnect
            [34,49]                                                        # Final exit
        ]

        for coords in maze_node_locs:
            self.add_node(node(coords[0]*scale, coords[1]*scale))

        for conn in maze_node_conns:
            self.add_connection(conn[0], conn[1])

        self.bounds = ((scale * 5, scale * 6), (-scale * 4, -scale * 1))
        self.update_node_number_and_connections()

    
