import pandas as pd
import numpy as np

# Path to dataset
datapath = "dataset.csv"
# DataFrame columns' names
colnames = np.array(["Age", "Gender", "tBilirubin", "dBilirubin", "tProteins", "Albumin", " A/G", "SGPT", "SGOT", "Alkphos", "Label"])

# Read csv file and convert it to pandas DataFrame
data = pd.read_csv(datapath, names=colnames)

# Prints created dataframe to debug #debug
print(data) #debug

# Change values from "Male" and "Female" to "0" and "1"
for i in range(data.shape[0]):
    if data.at[i, "Gender"] == "Male":
        data.at[i, "Gender"] = 0
    elif data.at[i, "Gender"] == "Female":
        data.at[i, "Gender"] = 1

# Change healthy person's label to 0 (Not having disease) and not healthy to 1 (having the disease)
# (There is no big deal about that but it feels more professional when we are talking about classification)
    if data.at[i, "Label"] == 2:
        data.at[i, "Label"] = 0

# Print modified dataframe to debug #debug
print(data) #debug

# Convert Dataframe to Numpy Array
arr_data = data.to_numpy()
print(arr_data) #debug