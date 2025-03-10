import matplotlib.pyplot as plt
import numpy as np
import json
import os
import mplcursors
from main import AF, seeds, num_epochs


evalulated_data = {
    "top_3_accuracies": {},
    "top_3_times": {}, # (top min times)
    "top_3_losses": {}, # (top min losses)
    "top_3_l2_norms": {}, # (top min l2 norms)
    "top_3_conv_l2_norms": {}, # (top min conv l2 norms)
    "top_3_fc_l2_norms": {}, # (top min fc l2 norms)
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
                    file_path = os.path.join(activation_path, (file_str_with_k + f"_{num_epochs}.json"))
                    load_data_from_file(file_path)
            else:
                file_path = os.path.join(activation_path, (file_str + f"_{num_epochs}.json"))
                load_data_from_file(file_path)
        eval_data()
        plot_data(f"{i:03d}")
        save_data(f"MNIST/evaluated_data_{i:03d}_{num_epochs}.json")
        data.clear()


def eval_data():
    data_objects = []
    for key in data:
        # MNIST/{activation}/{metrics}_{seed}_{num_epochs}.json for non B_SPLINE activations
        # MNIST/{activation}/{metrics}_{seed}_k{k}_{num_epochs}.json for B_SPLINE activations
        data_obj = {
            "Model": data[key]["activation"],
            "Accuracy": data[key]["test_accuracies"][-1],
            "Time": data[key]["train_times"][-1],
            "Loss": data[key]["train_losses"][-1],
            "L2 Norm": data[key]["l2_norms"][-1],
            "Conv L2 Norm": data[key]["conv_l2_norms"][-1],
            "FC L2 Norm": data[key]["fc_l2_norms"][-1],
            "seed": f"{seeds.index(data[key]['seed']):03d}",
            "k": data[key]["k"] if "k" in key else None
        }
        data_objects.append(data_obj)
    
    # sort accuracies (descending), times (ascending), losses (ascending)
    sorted_accuracies = np.argsort([-data_obj["Accuracy"] for data_obj in data_objects])
    sorted_times = np.argsort([data_obj["Time"] for data_obj in data_objects])
    sorted_losses = np.argsort([data_obj["Loss"] for data_obj in data_objects])
    l2_norms = np.argsort([data_obj["L2 Norm"] for data_obj in data_objects])
    conv_l2_norms = np.argsort([data_obj["Conv L2 Norm"] for data_obj in data_objects])
    fc_l2_norms = np.argsort([data_obj["FC L2 Norm"] for data_obj in data_objects])
    
    # top 3 accuracies, times, losses
    for i in range(3):
        evalulated_data["top_3_accuracies"][f"{i+1}."] = data_objects[sorted_accuracies[i]]
        evalulated_data["top_3_times"][f"{i+1}."] = data_objects[sorted_times[i]]
        evalulated_data["top_3_losses"][f"{i+1}."] = data_objects[sorted_losses[i]]
        evalulated_data["top_3_l2_norms"][f"{i+1}."] = data_objects[l2_norms[i]]
        evalulated_data["top_3_conv_l2_norms"][f"{i+1}."] = data_objects[conv_l2_norms[i]]
        evalulated_data["top_3_fc_l2_norms"][f"{i+1}."] = data_objects[fc_l2_norms[i]]


def plot_data(seed_str):
    num_epochs = data[list(data.keys())[0]]["num_epochs"]
    accuracies = {}
    losses = {}
    times = {}
    l2_norms = {}
    conv_l2_norms = {}
    fc_l2_norms = {}
    for key in data:
        accuracies[key] = {
            "model": data[key]["activation"],
            "k": data[key]["k"] if "k" in key else None,
            "accuracies": data[key]["test_accuracies"],
            }
        losses[key] = {
            "model": data[key]["activation"],
            "k": data[key]["k"] if "k" in key else None,
            "losses": data[key]["train_losses"],
            }
        times[key] = {
            "model": data[key]["activation"],
            "k": data[key]["k"] if "k" in key else None,
            "times": data[key]["train_times"],
            }
        l2_norms[key] = {
            "model": data[key]["activation"],
            "k": data[key]["k"] if "k" in key else None,
            "l2_norms": data[key]["l2_norms"],
            }
        conv_l2_norms[key] = {
            "model": data[key]["activation"],
            "k": data[key]["k"] if "k" in key else None,
            "l2_norms": data[key]["conv_l2_norms"],
            }
        fc_l2_norms[key] = {
            "model": data[key]["activation"],
            "k": data[key]["k"] if "k" in key else None,
            "l2_norms": data[key]["fc_l2_norms"],
            }
    
    # plot data
    plt.figure(figsize=(24, 14))
    plt.suptitle(f"{seed_str} models compared")
    
    # plot accuracies
    ax1 = plt.subplot(3, 2, 1)
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
    ax2 = plt.subplot(3, 2, 2)
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
    ax3 = plt.subplot(3, 2, 3)
    ax3.set_title("Times")
    for key in times:
        label = f"{times[key]['model']}_k{times[key]['k']}" if times[key]["k"] else times[key]['model']
        ax3.plot(range(1, num_epochs + 1), times[key]["times"], label=label)
    ax3.grid()
    ax3.set_xlabel("Epochs")
    ax3.set_xticks(range(1, num_epochs + 1))
    ax3.set_ylabel("Time (s)")
    
    # plot l2 norms
    ax4 = plt.subplot(3, 2, 4)
    ax4.set_title("L2 Norms")
    for key in l2_norms:
        label = f"{l2_norms[key]['model']}_k{l2_norms[key]['k']}" if l2_norms[key]["k"] else l2_norms[key]['model']
        ax4.plot(range(1, num_epochs + 1), l2_norms[key]["l2_norms"], label=label)
    ax4.grid()
    ax4.set_xlabel("Epochs")
    ax4.set_xticks(range(1, num_epochs + 1))
    ax4.set_ylabel("L2 Norm")
    
    # plot conv l2 norms
    ax5 = plt.subplot(3, 2, 5)
    ax5.set_title("Conv L2 Norms")
    for key in conv_l2_norms:
        label = f"{conv_l2_norms[key]['model']}_k{conv_l2_norms[key]['k']}" if conv_l2_norms[key]["k"] else conv_l2_norms[key]['model']
        ax5.plot(range(1, num_epochs + 1), conv_l2_norms[key]["l2_norms"], label=label)
    ax5.grid()
    ax5.set_xlabel("Epochs")
    ax5.set_xticks(range(1, num_epochs + 1))
    ax5.set_ylabel("Conv L2 Norm")
    
    # plot fc l2 norms
    ax6 = plt.subplot(3, 2, 6)
    ax6.set_title("FC L2 Norms")
    for key in fc_l2_norms:
        label = f"{fc_l2_norms[key]['model']}_k{fc_l2_norms[key]['k']}" if fc_l2_norms[key]["k"] else fc_l2_norms[key]['model']
        ax6.plot(range(1, num_epochs + 1), fc_l2_norms[key]["l2_norms"], label=label)
    ax6.grid()
    ax6.set_xlabel("Epochs")
    ax6.set_xticks(range(1, num_epochs + 1))
    ax6.set_ylabel("FC L2 Norm")
    
    # adjust subplots
    plt.subplots_adjust(hspace=0.4)
    mplcursors.cursor(hover=True)
    
    # save as png
    plt.savefig(f"MNIST/evaluated_{seed_str}_{num_epochs}.png")
        

def save_data(path="MNIST/evaluated_data.json"):
    with open(path, "w") as file:
        json.dump(evalulated_data, file)
    


if __name__ == "__main__":
    load_data()
