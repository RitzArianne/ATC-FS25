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
    max_time : float = 5.0

    # Animation
    draw_trajectories : bool = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--num_agents", type = int, default= default_vals.num_agents, help= "number of indipendant agents/drones/robots in the simulation")
    #parser.add_argument("-d", "--draw_traj", type= bool, default= default_vals.draw_trajectories, help= "True: singular image with trajectories, False: video")
    parser.add_argument("-m", "--map", type = str, default=default_vals.map, help= "select from: loss, ...")
    parser.add_argument("-t", "--max_time", type = float, default= default_vals.max_time, help= "set simulation end time")


    args = parser.parse_args()
    # Map Init :
    if (args.map == "loss") :
        simulated_map = loss_map()
        #simulated_map.print_map()
    else:
        print(f"Not Found: You entered {args.map}")

    run(num_agents= args.num_agents, max_time= args.max_time, map= simulated_map)

    print("DONE")





