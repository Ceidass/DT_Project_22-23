import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # Question 1c

# Path to dataset
datapath = "dataset.csv"
# DataFrame columns' names
colnames = np.array(["Age", "Gender", "tBilirubin", "dBilirubin", "tProteins", "Albumin", "A/G", "SGPT", "SGOT", "Alkphos", "Label"])

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

###################################################### Question 1c ####################################################################

# Creating Standard Scaler object
stdsclr = MinMaxScaler(feature_range=(-1, 1))

# Transforms the selected columns' data to be normalized to range [-1,1]
data[["Age", "tBilirubin", "dBilirubin",
"tProteins", "Albumin", "A/G", "SGPT", "SGOT", "Alkphos"]] = stdsclr.fit_transform(data[["Age", "tBilirubin", "dBilirubin",
"tProteins", "Albumin", "A/G", "SGPT", "SGOT", "Alkphos"]])
print(data) # debug

# Converting DataFrame to Numpy Array
arr_data = data.to_numpy()
print(arr_data) # debug

# Split array into Features (first 10 columns) and Labels (last column)
features ,labels = np.hsplit(arr_data,[10])
labels = labels.flatten() # Flatten array to be 1-Dimensional
print("Features")
print(features.shape)
print("Labels")
print(labels.shape)