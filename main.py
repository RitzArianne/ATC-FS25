import argparse
import numpy as np

from typing import List, Tuple

from environment import loss_map, line_map, maze_map, complex_maze_map
from agents import agent
from physics import constants

import random

class default_vals: # Intended for Parser
    # Simulation
    num_agents : int = 1
    map : str = "loss"
    max_time : float = 60.0
    print_solver_feedback: bool = False

def run (num_agents : int, max_time : float, map : line_map):

    max_time_steps = int(max_time/constants.dt)
    save_data = np.empty((num_agents,max_time_steps + 1,4))

    # Initialize Agents
    agents_list : List[agent] = []
    for i in range(num_agents):
        agents_list.append(agent(map, np.array([0,0])))
        agents_list[i].name = f"Agent with index: {i}"
        save_data[i,0,:] = np.resize(agents_list[i].W_p_COM,(4,))

    step : int = 1
    while(step <= max_time_steps):
        # Agent Communication
        adverts: List[Tuple[float, Tuple[float, float]]] = []
        for agents in agents_list:
            adverts.append(agents.advertise())

        # Agent Evaluation
        for i, agents in enumerate(agents_list):
            try: 
                agents.update(agents.find_input(agents.find_first_intermediate_target().to_numpy()), adverts=adverts[:i] + adverts[i+1:])
                save_data[i, step, :] = np.resize(agents.W_p_COM,(4,))
            except Exception as e:
                agents.physics_step() # Maybe this helps seeing what went wrong
                save_data[i, step, :] = np.resize(agents.W_p_COM,(4,))
                np.save("np_saves/last_run", save_data) # make sure data is saved in case of error
                raise AssertionError(f"ran into {e}")
            
        step += 1

    np.save("np_saves/last_run", save_data)
    print("\n--- RUN DONE ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",   "--num_agents", type = int,     default= default_vals.num_agents,           help= "number of indipendant agents/drones/robots in the simulation")
    parser.add_argument("-m",   "--map",        type = str,     default=default_vals.map,                   help= "select from: loss, maze, c_maze")
    parser.add_argument("-t",   "--max_time",   type = float,   default= default_vals.max_time,             help= "set simulation end time")

    args = parser.parse_args()
    # Map Init
    match args.map:
        case "loss":
            simulated_map = loss_map()  
        case "maze":
            simulated_map = maze_map()
        case "c_maze":
            simulated_map = complex_maze_map()
        case _:
            print(f"Map not Found: You entered {args.map}")
            simulated_map = loss_map()

    # Assign Goal Node here:
    goal_node_idx: int = round(random.random() * (len(simulated_map.nodes) - 1))
    simulated_map.nodes[goal_node_idx].name = "GOAL Node"
    print(f"selected Goal Node is {simulated_map.nodes[goal_node_idx]}")
    #simulated_map.print_map()

    run(num_agents= args.num_agents, max_time= args.max_time, map= simulated_map)

    print("DONE")