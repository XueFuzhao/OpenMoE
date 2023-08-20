import csv
import pandas as pd

csv_file_path = '../../final_results/BIG-bench-Lite/retrieved_results.csv'

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

csv_dataframe = read_csv_and_convert_to_dataframe(csv_file_path)

# print(csv_dataframe['winowhy'][0])
for column in csv_dataframe.columns:
    csv_dataframe[column] = csv_dataframe[column].apply(lambda x: [float(value) for value in x.split()][0] if " " in x else x)

# Calculate the average value for each column
print(csv_dataframe)
# csv_dataframe['conlang_translation'] = csv_dataframe['conlang_translation'] / 100.0
csv_dataframe['row_mean']  = csv_dataframe.iloc[:, 1:].mean(axis=1)
print(csv_dataframe)


