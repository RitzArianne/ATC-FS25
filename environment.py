from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from numpy.typing import NDArray

class node () :
    x : float = 0
    y : float = 0
    name : str = "No Name"
    node_idx : int
    connectivity : List[int] = []

    def __init__(self, x:float, y:float) :
        self.x = x
        self.y = y

    def change_name(self, new_name : str) :
        self.name = new_name


class line_map () :
    nodes : list[node] = []
    connections : List[Tuple[int, int]] = []
    bounds : Tuple[Tuple[int,int]]
    

    def add_node(self, new_node : node) :
        # TODO: add minimum di
        # stance requirement
        self.nodes.append(new_node)

    def add_connection(self, node_1 : int, node_2 : int) :
        if (len(self.nodes) > max(node_1, node_2) and node_1 >= 0 and node_2 >= 0):
            self.connections.append((node_1, node_2))
        else:
            print(f"Problem adding [{node_1}, {node_2}] to connections when highest idx node is {len(self.nodes) - 1}")

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

        for idx, pos_i in enumerate(agent_list):
            plt.plot(pos_i[0], pos_i[0], "H", c = 'green', label=f"Agent: {idx}")
    
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Map with Nodes and Connections')
        plt.legend()
        plt.grid(True)
        plt.show()
        


class loss_map(line_map):
    def __init__ (self, scale : float = 1.0):
        loss_node_loc = [[0,0],[0,1],[0,-1],[1.0/3,0],[1.0/3,1],[2.0/3,0],[2.0/3,1],[1,0],[1.0/3,-0.5],[1.0/3,-1],[1,-0.5],[-1.0/3,0],[-0.5,0],[-2.0/3,0],[-1,0],[-1.0/3,-1],[-0.5,1],[-2.0/3,-1]]
        loss_node_con = [[0,1],[0,2],[0,3],[3,4],[3,5],[3,8],[5,6],[5,7],[8,10],[8,9],[0,11],[11,15],[11,12],[12,16],[12,13],[13,14],[13,17]]

        for coords in loss_node_loc:
            self.add_node(node(coords[0]*scale, coords[1]*scale))

        for line in loss_node_con:
            self.add_connection(line[0], line[1])

        self.bounds = ((scale * 1,scale * 1), (scale *-1, scale * -1))

        self.update_node_number_and_connections()
    
