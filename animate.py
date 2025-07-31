import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
import argparse

color_map = ['b', 'g', 'r', 'c', 'm', 'y']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="last_run", help="pass the name of wanted file for animation")
    args = parser.parse_args()

    # Load the data
    data = np.load(f"np_saves/{args.name}.npy")  # Shape: (num_agents, num_time_steps, 2)
    num_agents, num_time_steps, _ = data.shape

    # Set up the figure and axes
    fig, ax = plt.subplots()
    lines = []
    for i in range(num_agents):
        (line,) = ax.plot([], [], color=color_map[i % len(color_map)], label=f'Agent {i}')
        lines.append(line)

    ax.set_xlim(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
    ax.set_ylim(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
    ax.set_xlabel('X-axis label')
    ax.set_ylabel('Y-axis label')
    ax.set_title('Agent Animation')
    ax.legend()

    # Animation update function
    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(data[i, :frame+1, 0], data[i, :frame+1, 1])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=num_time_steps, interval=50, blit=True)

    # Save or show animation
    # ani.save("agent_animation.mp4", writer='ffmpeg')  # Optional: Save to file
    plt.show()

    print("-- animation done --")