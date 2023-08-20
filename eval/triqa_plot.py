import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data = [
    ("OPT", 7, 7, 300, 2.1, 22.7),
    ("OPT", 13, 13, 300, 3.9, 28.2),
    ("Pythia", 7, 7, 300, 2.1, 19.8),
    ("Pythia", 12, 12, 300, 3.6, 22.3),
    ("GPTJ", 6, 6, 400, 2.4, 23.4),
    ("GPT-NeoX", 20, 20, 475, 9.5, 34.7),
    ("MPT", 7, 7, 1000, 7, 34.3),
    ("LLaMA", 7, 7, 1000, 7, 44.3),
    ("GaLM-Dense", 0.1, 0.1, 600, 0.6, 2.3),
    ("GaLM-Dense", 2, 2, 600, 1.2, 27),
    ("GaLM-Dense", 8, 8, 600, 4.8, 48.1),
    ("GaLM-MoE", 0.1, 2, 600, 0.6, 9.4),
    ("GaLM-MoE", 2, 27, 600, 1.2, 44),
    ("GaLM-MoE", 8, 143, 600, 4.8,  55.1),
    ("Gopher", 1, 1, 300, 0.3, 6.5),
    ("Gopher", 7, 7, 300, 2.1, 19.9),
    ("PaLM", 8, 8, 780, 6.2, 39.5),
    ("GPT-3", 3, 3, 300, 0.9, 31.3),
    ("GPT-3", 7, 7, 300, 2.1, 38.7),
    ("GPT-3", 13, 13, 300, 3.9, 41.8),
    ("OpenMoE", 0.2, 0.5, 200, 0.04, 12.8),
    ("OpenMoE", 2, 8, 200, 0.4, 29.2),
]

# Group data by model name and calculate the average values
model_data = {}
for model, act_param, total_param, tokens, cost, result in data:
    if model not in model_data:
        model_data[model] = {"cost": [], "result": [], "total_param": [], "act_param": []}
    model_data[model]["cost"].append(cost)
    model_data[model]["result"].append(result)
    model_data[model]["total_param"].append(total_param)
    model_data[model]["act_param"].append(act_param)

# Plotting
plt.figure(figsize=(12, 6))
colors = plt.cm.get_cmap("tab20", len(model_data))

handles = []
for i, (model, values) in enumerate(model_data.items()):
    x = values["cost"]
    y = values["result"]
    sizes = [param * 20 for param in values["act_param"]]  # Adjust the scaling factor as needed
    total_sizes = [param * 20 for param in values["total_param"]]
    color = 'red' if model == "OpenMoE" else colors(i)
    marker= 'o'
    if "MoE" in model:
        plt.scatter(x, y, label=model, color='lightgray', s=total_sizes, marker=marker)
    plt.scatter(x, y, label=model, color=color, s=sizes, marker=marker)
    # handle = mpatches.Patch(color=colors(i), label=f'{model}', marker='o')
    handle = Line2D([0], [0], marker=marker, color='w', label=f'{model}', markersize=12,
           markerfacecolor=color)
    handles.append(handle)
    # Adding dashed lines to connect dots of the same model
    plt.plot(x, y, linestyle='dashed', color=color)
    # Annotate each dot with the model name and total parameters
    for j, (x_val, y_val) in enumerate(zip(x, y)):
        plt.annotate(f'{model}-{values["total_param"][j]}B', (x_val, y_val), textcoords="offset points", xytext=(0,10), ha='left', size=7)

plt.xlabel("Relative Cost")
plt.ylabel("TrivalQA (0-shot EM)")
plt.title("Relative Cost vs TrivalQA (0-shot EM)")
# Move the legend to the outside right of the main figure
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.10, 1), title="Model")
plt.subplots_adjust(right=0.75)
plt.grid(True)
# plt.show()
plt.savefig("../figure/triqa.pdf", dpi=300, bbox_inches="tight")
plt.savefig("../figure/triqa.png", dpi=300, bbox_inches="tight")
