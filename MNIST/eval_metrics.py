import matplotlib.pyplot as plt
import numpy as np
import json
import os
import mplcursors
from main import AF, seeds


evalulated_data = {
    "top_3_accuracies": {},
    "top_3_times": {}, # (top min times)
    "top_3_losses": {}, # (top min losses)
}

data = {}

def load_data_from_file(file_path):
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as file:
        data[f"{file_path}"] = json.load(file)


def load_data():
    path = "MNIST"
    
    for i in range(len(seeds)):
        file_str = f"metrics_{i:03d}"
        for activation in AF:
            activation_path = os.path.join(path, activation.name)
            if not os.path.exists(activation_path):
                continue
            activation_path = os.path.join(activation_path, "metrics")
            if "B_SPLINE" in activation.name:
                for j in range(2): # k=2 and k=3
                    file_str_with_k = f"{file_str}_k{j+2}"
                    file_path = os.path.join(activation_path, (file_str_with_k + ".json"))
                    load_data_from_file(file_path)
            else:
                file_path = os.path.join(activation_path, (file_str + ".json"))
                load_data_from_file(file_path)
        eval_data()
        plot_data(f"{i:03d}")
        save_data(f"MNIST/evaluated_data_{i:03d}.json")
        data.clear()


def eval_data():
    data_objects = []
    for key in data:
        data_obj = {
            "Model": key.split("/")[1],
            "Accuracy": data[key]["test_accuracies"][-1],
            "Time": data[key]["train_times"][-1],
            "Loss": data[key]["train_losses"][-1],
            "seed": key.split("_")[-1].split(".")[0],
            "k": key.split("_")[-1].split("k")[-1] if "k" in key else None
        }
        data_objects.append(data_obj)
    
    # sort accuracies (descending), times (ascending), losses (ascending)
    sorted_accuracies = np.argsort([-data_obj["Accuracy"] for data_obj in data_objects])
    sorted_times = np.argsort([data_obj["Time"] for data_obj in data_objects])
    sorted_losses = np.argsort([data_obj["Loss"] for data_obj in data_objects])
    
    # top 3 accuracies, times, losses
    for i in range(3):
        evalulated_data["top_3_accuracies"][f"{i+1}."] = data_objects[sorted_accuracies[i]]
        evalulated_data["top_3_times"][f"{i+1}."] = data_objects[sorted_times[i]]
        evalulated_data["top_3_losses"][f"{i+1}."] = data_objects[sorted_losses[i]]


def plot_data(seed_str):
    num_epochs = data[list(data.keys())[0]]["num_epochs"]
    accuracies = {}
    losses = {}
    times = {}
    for key in data:
        accuracies[key] = {
            "model": key.split("/")[1],
            "k": key.split("_")[-1].split("k")[-1] if "k" in key else None,
            "accuracies": data[key]["test_accuracies"],
            }
        losses[key] = {
            "model": key.split("/")[1],
            "k": key.split("_")[-1].split("k")[-1] if "k" in key else None,
            "losses": data[key]["train_losses"],
            }
        times[key] = {
            "model": key.split("/")[1],
            "k": key.split("_")[-1].split("k")[-1] if "k" in key else None,
            "times": data[key]["train_times"],
            }
    
    # plot data
    plt.figure(figsize=(24, 14))
    plt.suptitle(f"{seed_str} models compared")
    
    # plot accuracies
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title("Accuracies")
    for key in accuracies:
        label = f"{accuracies[key]['model']}_k{accuracies[key]['k']}" if accuracies[key]["k"] else accuracies[key]['model']
        ax1.plot(range(1, num_epochs + 1), accuracies[key]["accuracies"], label=label)
    ax1.grid()
    ax1.set_xlabel("Epochs")
    ax1.set_xticks(range(1, num_epochs + 1))
    ax1.set_ylabel("Accuracy (%)")
    mplcursors.cursor(hover=True)
    
    
    # lässt die legende neben dem ersten subplot erscheinen
    leg = plt.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    leg.set_draggable(True)
    leg.set_title("Models")
    
    # plot losses
    ax2 = plt.subplot(3, 1, 2)
    ax2.set_title("Losses")
    for key in losses:
        label = f"{losses[key]['model']}_k{losses[key]['k']}" if losses[key]["k"] else losses[key]['model']
        ax2.plot(range(1, num_epochs + 1), losses[key]["losses"], label=label)
    ax2.grid()
    ax2.set_xlabel("Epochs")
    ax2.set_xticks(range(1, num_epochs + 1))
    ax2.set_ylabel("Loss")
    mplcursors.cursor(hover=True)
    
    # plot times
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title("Times")
    for key in times:
        label = f"{times[key]['model']}_k{times[key]['k']}" if times[key]["k"] else times[key]['model']
        ax3.plot(range(1, num_epochs + 1), times[key]["times"], label=label)
    ax3.grid()
    ax3.set_xlabel("Epochs")
    ax3.set_xticks(range(1, num_epochs + 1))
    ax3.set_ylabel("Time (s)")
    
    plt.subplots_adjust(hspace=0.4)
    mplcursors.cursor(hover=True)
    
    plt.savefig(f"MNIST/evaluated_{seed_str}.png")
    plt.show()
        

def save_data(path="MNIST/evaluated_data.json"):
    with open(path, "w") as file:
        json.dump(evalulated_data, file)
    


if __name__ == "__main__":
    load_data()
