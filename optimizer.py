import casadi as ca
import numpy as np
from typing import List
import pandas as pd

from agents import agent
from physics import constants
from environment import line_map

def run (num_agents : int, max_time : float, map : line_map):
    """
    dummy_data = np.array([num_agents])
    df = pd.DataFrame(dummy_data)
    df.to_csv("csv/dummy_data.csv")
    """
    
    """
    TODO: define each agents system dynamics
    TODO: define cost function
    TODO: define geometries
    TODO: anyting casadi
    TODO: save data to csv
    """

    agents_list : List[agent] = []
    for i in range(num_agents):
        agents_list.append(agent(map, np.zeros(3)))
        agents_list[i].name = f"Agent Nr. {i}"

    time : float = 0.0
    while(time <= max_time):
        for i in range(num_agents):
            agents_list[i].update()

        map.print_map_and_agents([agent_i.W_p_COM for agent_i in agents_list])
        #input(f"Press button to step forward, current time {time}")
        time += constants.dt