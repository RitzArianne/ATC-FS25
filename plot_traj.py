import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
import argparse

color_map = ['b', 'g', 'r', 'c', 'm', 'y']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="last_run", help="pass the name of wanted file for animation")

    args = parser.parse_args()

    # Load the data from the specified .npy file
    data = np.load(f"np_saves/{args.name}.npy")
    num_agents, num_time_steps, _ = data.shape

    for time in range(600):

        plt.figure()
        for i in range(num_agents):
            # Use the color map to get the color for each agent
            plt.scatter(data[i, time, 0], data[i, time, 1], color=color_map[i % len(color_map)], label=f'Agent {i}')

        plt.xlabel('X-axis label')  # Add appropriate labels
        plt.ylabel('Y-axis label')
        plt.title(f'Environment at time {time}')  # Add a title
        plt.legend()  # Show legend if needed
        plt.show()
    print("-- animation done --")