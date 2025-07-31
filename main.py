import argparse
import numpy as np
import pandas as pd

from typing import List, Tuple

from environment import loss_map, line_map, maze_map, complex_maze_map
from agents import agent, agent_parameters
from physics import constants

class default_vals: # Intended for Parser
    # Simulation
    num_agents : int = 1
    map : str = "loss"
    max_time : float = 60.0
    print_solver_feedback: bool = False

    # Animation
    draw_trajectories : bool = True

def run (num_agents : int, max_time : float, map : line_map, verbose: bool):

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
        #  print(f"all adverts are: {adverts}")
        for i, agents in enumerate(agents_list):
            try: 
                agents.update(agents.find_input(agents.find_first_intermediate_target().to_numpy(), print_solver=verbose), adverts=adverts[:i] + adverts[i+1:])
                save_data[i, step, :] = np.resize(agents.W_p_COM,(4,))
            except Exception as e:
                print(f"Agent BEFORE error {agents}")
                closeset_node, distance = agents.global_map.find_closest_node(agents.W_p_COM[0], agents.W_p_COM[1])
                print(f"distance to {closeset_node} is {distance}")
                agents.physics_step() # Maybe this helps seeing what went wrong
                print(f"Agent AFTER error (allowing it to continue movement) {agents}")
                closeset_node, distance = agents.global_map.find_closest_node(agents.W_p_COM[0], agents.W_p_COM[1])
                print(f"distance to {closeset_node} is {distance}")
                print(f"targetting: {agents.target_node}, at: {agents.current_node} with connectivity to {agents.current_node.connectivity}")
                save_data[i, step, :] = np.resize(agents.W_p_COM,(4,))
                np.save("np_saves/last_run", save_data)

                print(agents.A, agents.B)
                raise AssertionError(f"ran into {e}")

        # map.print_map_and_agents([agent_i.W_p_COM for agent_i in agents_list])

        #input(f"Press button to step forward, current time {time}")
        step += 1

    np.save("np_saves/last_run", save_data)
    print("\n--- RUN DONE ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",   "--num_agents", type = int,     default= default_vals.num_agents,           help= "number of indipendant agents/drones/robots in the simulation")
    # parser.add_argument("-d",  "--draw_traj",  type= bool,     default= default_vals.draw_trajectories,    help= "True: singular image with trajectories, False: video")
    parser.add_argument("-m",   "--map",        type = str,     default=default_vals.map,                   help= "select from: loss, maze, c_maze")
    parser.add_argument("-t",   "--max_time",   type = float,   default= default_vals.max_time,             help= "set simulation end time")
    parser.add_argument("-v",   "--verbose",    type = bool,    default= default_vals.print_solver_feedback,help= "print cvxpy solver for all agents and timesteps")

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
            print(f"Not Found: You entered {args.map}")
            simulated_map = loss_map()

    #simulated_map.print_map()
    run(num_agents= args.num_agents, max_time= args.max_time, map= simulated_map, verbose=args.verbose)

    print("DONE")





