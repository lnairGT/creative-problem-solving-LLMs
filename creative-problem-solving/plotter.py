import matplotlib.pyplot as plt
import numpy as np


def plot_results(mode, data):
    tasks = ["Scoop", "Hammer", "Spatula", "Toothpick", "Pliers", "Overall"]
    data = {k: list(v.values()) for k, v in data.items()}
    task_type = mode
    # Create a N_models x N_tasks grid
    N_tasks = len(tasks)
    N_models = len(data)
    grid_values = np.random.rand(N_models, N_tasks)
    for i in range(N_models):
        grid_values[i,:] = data[list(data.keys())[i]]

    # Plot the grid with shading
    cmap = {
        "nominal": "Reds",
        "creative": "Oranges",
        "creative-obj": "Purples",
        "creative-task": "Greens",
        "creative-task-obj": "Blues"
    }
    plt.imshow(grid_values, cmap=cmap[task_type], vmin=0, vmax=1)

    # Add text annotations for each cell
    for i in range(len(grid_values)):
        for j in range(len(grid_values[i])):
            if grid_values[i, j] > 0.5:
                plt.text(j, i, f'{grid_values[i, j]:.2f}', ha='center', va='center', color='white')
            else:
                plt.text(j, i, f'{grid_values[i, j]:.2f}', ha='center', va='center', color='black')

    # Add colorbar to show the mapping of values to shades
    plt.colorbar()

    # Customize x-axis ticks (columns)
    plt.xticks(range(len(grid_values[0])), tasks)
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True, rotation=45)

    # Customize y-axis ticks (rows)
    plt.yticks(range(len(grid_values)), data.keys())

    # Add a thick line separation between the last two columns
    plt.axvline(x=len(grid_values[0]) - 1.5, color='white', linewidth=3)

    if task_type == "creative":
        title = "Model performance with regular prompts"
    elif task_type == "creative-obj":
        title = "Model performance with object augmented prompts"
    elif task_type == "creative-task":
        title = "Model performance with task augmented prompts"
    elif task_type == "creative-task-obj":
        title = "Model performance with object and task augmented prompts"
    elif task_type == "nominal":
        title = "Model performance with regular prompts (no object replacement needed)"
    plt.title(title)

    plt.tight_layout()
    # Show the plot
    plt.savefig(f'Viz_{task_type}.png')
