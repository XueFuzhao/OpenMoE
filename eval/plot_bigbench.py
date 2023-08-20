import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


csv_file_path = '../../final_results/BIG-bench-Lite/retrieved_results.csv'
output_csv_file_dir = '../../final_results/BIG-bench-Lite'

model_mapping = {
    'GPT-3-3B': 'GPT-3_3B',
    'GPT-3-6B': 'GPT-3_6B',
    'GPT-3-13B': 'GPT-3_13B',
    'GPT-3-200B': 'GPT-3_200B',
    'GPT-3-Small': 'GPT-3_125m',
    'GPT-3-Medium': 'GPT-3_350m',
    'GPT-3-Large': 'GPT-3_760m',
    'GPT-3-XL': 'GPT-3_1300m',
    "all_examples": "OpenMoE_8B",
}

# Dictionary to update model sizes
model_size_updates = {
    'BIG-G-sparse_2m': 60630144,
    'BIG-G-sparse_16m': 234507776,
    'BIG-G-sparse_53m': 534215808,
    'BIG-G-sparse_125m': 1777677312,
    'BIG-G-sparse_244m': 2819788160,
    'BIG-G-sparse_422m': 4126141440,
    'BIG-G-sparse_1b': 7581906944,
    'BIG-G-sparse_2b': 17278886400,
    'BIG-G-sparse_4b': 25465853952,
    'BIG-G-sparse_8b': 60261322752,
}



def read_csv_and_convert_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("The CSV file could not be found.")
        return None
    except Exception as e:
        print("An error occurred:", str(e))
        return None

# Define a function to calculate num_tokens based on model name
def calculate_num_tokens(model_name):
    if model_name.startswith('BIG-G'):
        return 131 * 10**9
    elif model_name.startswith('PaLM'):
        return 780 * 10**9
    elif model_name.startswith('GPT-3'):
        return 300 * 10**9
    elif model_name.startswith('OpenMoE'):
        return 200 * 10**9
    else:
        return None

# Extract the model size information and convert it to desired format
def convert_size(size_str):
    print(size_str)
    size = int(size_str[:-1])
    unit = size_str[-1]
    if unit == 'm':
        return size * 10**6
    elif unit == 'b' or unit == 'B' :
        return size * 10**9
    else:
        return size


# Define a function to calculate activated_parameters based on model name
def calculate_activated_parameters(row):
    model_name = row['model']
    model_size = row['model_size']

    if model_name.startswith('OpenMoE'):
        model_activate_param_mapping = {
            'OpenMoE_8B': 2 * 10 ** 9
        }
        return model_activate_param_mapping.get(model_name, model_size)
    else:
        return model_size

# Read csv file
csv_dataframe = read_csv_and_convert_to_dataframe(csv_file_path)

# write one csv file for each shot
_number_of_shots = [3] # [0,1,2,3]
for shot in _number_of_shots:
    csv_dataframe_tmp = csv_dataframe.copy()
    for column in csv_dataframe_tmp.columns:
        csv_dataframe_tmp[column] = csv_dataframe_tmp[column].apply(
            lambda x: [float(value) for value in x.split()][shot] if " " in x else x)

    csv_dataframe_tmp['model'] = csv_dataframe_tmp['model'].replace(model_mapping)

    # Filter out rows with 'T=0' in 'model' column
    csv_dataframe_tmp = csv_dataframe_tmp[~csv_dataframe_tmp['model'].str.contains('T=0')]

    csv_dataframe_tmp['average'] = csv_dataframe_tmp.iloc[:, 1:].mean(axis=1)

    csv_dataframe_tmp['model_name'] = csv_dataframe_tmp['model'].apply(lambda x: x.split('_')[0])

    csv_dataframe_tmp['model_size'] = csv_dataframe_tmp['model'].str.extract(r'(\d+[mMbB])')[0].apply(convert_size)

    # Add 'activated_parameters' column using the function
    csv_dataframe_tmp['activated_parameters'] = csv_dataframe_tmp.apply(calculate_activated_parameters, axis=1)
    # Update 'model_size' column based on model names in the dictionary
    csv_dataframe_tmp.loc[csv_dataframe_tmp['model'].isin(model_size_updates.keys()), 'model_size'] = csv_dataframe_tmp['model'].map(model_size_updates)

    # Add 'num_tokens' column using the function
    csv_dataframe_tmp['num_tokens'] = csv_dataframe_tmp['model'].apply(calculate_num_tokens)


    csv_dataframe_tmp['cost'] = (csv_dataframe_tmp['activated_parameters']/(10.0**9)) * (csv_dataframe_tmp['num_tokens']/(10.0**9))
    csv_dataframe_tmp['cost'] = csv_dataframe_tmp['cost'] / 1000.0
    csv_dataframe_col = csv_dataframe_tmp[
        [
            'model',
            'model_name',
            'model_size',
            'activated_parameters',
            'num_tokens',
            'cost',
            'average'
         ]
    ]
    # print(csv_dataframe_col)
    file_to_save = f'{output_csv_file_dir}/{shot}-shot.csv'
    csv_dataframe_col.to_csv(
        file_to_save, header=True, index=False)



# marker_shapes = {
#     'BIG-G': 's',
#     'GPT-3': 's',
#     'BIG-G-sparse': 'o',
#     'OpenMoE': 'o',
# }


# Plot and save the figures
for shot in _number_of_shots:
    # read csv file again
    csv_file_path = f'{output_csv_file_dir}/{shot}-shot.csv'
    df = read_csv_and_convert_to_dataframe(csv_file_path)
    # print(df)
    df['cost'] = df['cost'].astype(float)
    df['average'] = df['average'].astype(float)
    df['activated_parameters'] = df['activated_parameters'].astype(float) / 1000000000.0
    df['model_size'] = df['model_size'].astype(float) / 1000000000.0
    df = df[0.05 < df['cost']][df['cost'] < 1.5]
    # df.sort_values(by=['model_name', 'activated_parameters'])

    model_data = {}
    # Convert the DataFrame to a dictionary and store it in the data_dict
    for model_name, group_tmp in df.groupby('model_name'):
        if model_name not in model_data:
            model_data[model_name] = {"cost": [], "result": [], "total_param": [], "act_param": []}
        # Append each row of the group to the corresponding lists
        # print(model_data[model_name])
        for index, row in group_tmp.sort_values(by=['cost']).iterrows():
            model_data[model_name]["cost"].append(row["cost"])
            model_data[model_name]["result"].append(row['average'])
            # print(float(row['model_size']), float(row['model_size'])/1000000000.0)
            model_data[model_name]["total_param"].append(row['model_size'])
            model_data[model_name]["act_param"].append(row['activated_parameters'])

    # Plotting
    plt.figure(figsize=(12, 6))
    colors = plt.cm.get_cmap("viridis", len(model_data))

    handles = []
    for i, (model, values) in enumerate(model_data.items()):
        x = values["cost"]
        y = values["result"]
        # print(values["act_param"])
        sizes = [param * 20.0 for param in values["act_param"]]  # Adjust the scaling factor as needed
        total_sizes = [param * 20.0 for param in values["total_param"]]
        color = 'red' if model == "OpenMoE" else colors(i)
        marker = 'o'  # if "MoE" in model else 'o'
        print(model)
        if ("MoE" in model ) or ('sparse' in model):
            print(model)
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
            if values["total_param"][j] >= 1.0:
                dot_name = f'{model}-{int(values["total_param"][j])}B'
            else:
                dot_name = f'{model}-{int(values["total_param"][j]*100)}M'
            plt.annotate(dot_name, (x_val, y_val), textcoords="offset points",
                         xytext=(0, 10), ha='left', size=7)

    plt.xlabel("Relative Cost")
    plt.ylabel(f"BigBench-Lite ({shot}-shot)")
    plt.title(f"Relative Cost vs BigBench-Lite ({shot}-shot)")
    # Move the legend to the outside right of the main figure
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.10, 1), title="Model")
    plt.subplots_adjust(right=0.75)
    plt.grid(True)
    # plt.show()
    plt.savefig(f"../figure/bblite-{shot}-shot.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"../figure/bblite-{shot}-shot.png", dpi=300, bbox_inches="tight")



    # df = df.sort_values(by=['model_name', 'activated_parameters'])
    # # Calculate the size of the dots based on activated parameters
    # df['dot_size'] = df['activated_parameters']/1000000000
    # # Get the unique model names
    # unique_model_names = df['model_name'].unique()
    # # Create a color palette based on unique model names
    # color_palette = sns.color_palette("Set1", n_colors=len(df.groupby('model_name')))
    # print(color_palette)
    # # Set style using Seaborn
    # sns.set(style="whitegrid")
    # # Create the plot
    # plt.figure(figsize=(10, 6))
    # # Scatter plot for individual data points
    # sns.scatterplot(
    #     x='cost',
    #     y='average',
    #     hue='model_name',
    #     palette=color_palette,
    #     size='dot_size',  # Use dot size based on activated parameters
    #     # sizes=(20, 200),  # Define the range of dot sizes
    #     style='model_name',
    #     markers=marker_shapes,
    #     data=df
    # )
    #
    # # Line plot to connect models with the same model_name
    # for model_name, group_tmp in df.groupby('model_name'):
    #     group = group_tmp.sort_values(by=['cost'])
    #     sns.lineplot(
    #         x='cost',
    #         y='average',
    #         data=group,
    #         color=color_palette[unique_model_names.tolist().index(model_name)],
    #         dashes=True #  if model_name.startswith('BIG-G') else False  # Use dashed line for models with specific names
    #     )
    # for i, (model_name, group_tmp) in enumerate(df.groupby('model_name')):
    #     group = group_tmp.sort_values(by=['cost'])
    #     sns.scatterplot(
    #         x='cost',
    #         y='average',
    #         hue='model_name',
    #         color_palette=color_palette[i:i+1],
    #         style='model_name',
    #         markers=marker_shapes,
    #         data=group
    #     )
        # plt.plot(
        #     group['cost'],
        #     group['average'],
        #     markers=marker_shapes,
        #     linestyle='--',
        #     label=model_name
        # )

    # plt.xlabel('Cost')
    # plt.ylabel('Average')
    # plt.legend()
    # plt.title('Average vs. Cost by Model')
    # plt.grid(True)
    # plt.show()
    # print(asd)

