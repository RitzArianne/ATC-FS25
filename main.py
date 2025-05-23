import argparse
import typing
import numpy as np

from optimizer import run
import environment
from environment import loss_map

class default_vals:
    # Simulation
    num_agents : int = 1
    map : str = "loss"

    # Animation
    draw_trajectories : bool = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--num_agents", type = int, default= default_vals.num_agents, help= "number of indipendant agents/drones/robots in the simulation")
    #parser.add_argument("-t", "--draw_traj", type= bool, default= default_vals.draw_trajectories, help= "True: singular image with trajectories, False: video")
    parser.add_argument("-m", "--map", type = str, default=default_vals.map, help= "select from: loss, ...")


    args = parser.parse_args()
    # Map Init :
    if (args.map == "loss") :
        simulated_map = loss_map()
        simulated_map.print_map()
    else:
        print(f"Not Found: You entered {args.map}")

    print("DONE")

    #run(args.num_agents)





