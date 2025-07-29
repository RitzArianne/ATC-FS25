import argparse
import numpy as np
import pandas as pd

from typing import List, Tuple

from environment import loss_map, line_map
from agents import agent, agent_parameters
from physics import constants

class default_vals:
    # Simulation
    num_agents : int = 1
    map : str = "loss"
    max_time : float = 20.0

    # Animation
    draw_trajectories : bool = True

def run (num_agents : int, max_time : float, map : line_map):
    """
    TODO: define geometries
    TODO: anyting casadi/cvxpy
    TODO: save data to csv
    """
    max_time_steps = int(max_time/constants.dt)
    save_data = np.empty((num_agents,max_time_steps + 1,4))

    agents_list : List[agent] = []
    for i in range(num_agents):
        agents_list.append(agent(map, np.array([0,0])))
        agents_list[i].name = f"Agent with index: {i}"
        save_data[i,0,:] = np.resize(agents_list[i].W_p_COM,(4,))

    step : int = 1
    while(step <= max_time_steps):
        for i, agents in enumerate(agents_list):
            agents.update(agents.find_input(agents.target_node.to_numpy(), verbose=False))
            print(agents)
            save_data[i, step, :] = np.resize(agents.W_p_COM,(4,))

        #map.print_map_and_agents([agent_i.W_p_COM for agent_i in agents_list])

        #input(f"Press button to step forward, current time {time}")
        step += 1

    np.save("np_saves/last_run", save_data)
    print("\n--- RUN DONE ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_agents", type = int,   default= default_vals.num_agents,        help= "number of indipendant agents/drones/robots in the simulation")
    #parser.add_argument("-d", "--draw_traj", type= bool,   default= default_vals.draw_trajectories, help= "True: singular image with trajectories, False: video")
    parser.add_argument("-m", "--map",        type = str,   default=default_vals.map,                help= "select from: loss")
    parser.add_argument("-t", "--max_time",   type = float, default= default_vals.max_time,          help= "set simulation end time")


    args = parser.parse_args()
    # Map Init :
    match args.map:
        case "loss":
            simulated_map = loss_map()
            #simulated_map.print_map()
        case _:
            print(f"Not Found: You entered {args.map}")
            simulated_map = loss_map()

    run(num_agents= args.num_agents, max_time= args.max_time, map= simulated_map)

    print("DONE")





