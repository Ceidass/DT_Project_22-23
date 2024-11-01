import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # Question 1c
from sklearn.neighbors import KNeighborsClassifier #Question 4b
from sklearn.model_selection import KFold #Question 4

# Path to dataset
datapath = "dataset.csv"
# DataFrame columns' names
colnames = np.array(["Age", "Gender", "tBilirubin", "dBilirubin", "tProteins", "Albumin", "A/G", "SGPT", "SGOT", "Alkphos", "Label"])

# Read csv file and convert it to pandas DataFrame
data = pd.read_csv(datapath, names=colnames)

# Prints created dataframe to debug #debug
#print(data) #debug

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


####### FIX MISSING VALUES #######
# Pointers to Nan values if exist
rowptr, colptr = np.where(data.isna())

# Iterate through column pointers
for i in range(colptr.shape[0]):
    if colptr[i] != 1 and colptr[i] != 1: # If we are not in Age's or Label's column
        # Replace the Nan values with the mean of that column
        data[data.columns[colptr[i]]].fillna(value=data[data.columns[colptr[i]]].mean(), inplace=True)
    else: # If we are in the labels column
        # Replace the Nan values with 0 or 1 randomly
        data[data.columns[colptr[i]]].fillna(value=np.random.randint(low=0, high=1), inplace=True)

# Print modified dataframe to debug #debug
#print(data) #debug

###################################################### Question 1c ################################################################

# Creating Standard Scaler object
stdsclr = MinMaxScaler(feature_range=(-1, 1))

# Transforms the selected columns' data to be normalized to range [-1,1]
data[["Age", "tBilirubin", "dBilirubin",
"tProteins", "Albumin", "A/G", "SGPT", "SGOT", "Alkphos"]] = stdsclr.fit_transform(data[["Age", "tBilirubin", "dBilirubin",
"tProteins", "Albumin", "A/G", "SGPT", "SGOT", "Alkphos"]])

# Print modified dataframe (with normalized columns) to debug #debug
#print(data) # debug

# Converting DataFrame to Numpy Array
arr_data = data.to_numpy(dtype=float)

# Split array into Features (first 10 columns) and Labels (last column)
features ,labels = np.hsplit(arr_data,[10])
labels = labels.flatten() # Flatten array to be 1-Dimensional

###################################################### Question 4b #################################################################

# List to store model average geometric mean, sensitivity and specificity

avg_geo = []
avg_sens = []
avg_spec = []
avg_acc = []


for k in range(3,16):
    print("K=%d" %(k)) #debug
    # List to store model accuracy, sensitivity and specificity of each Fold
    accuracy = []
    sens = []
    spec = []

    kf = KFold(n_splits=5)
    for i, (train_ind, test_ind) in enumerate(kf.split(features)):
        # Create KNN model
        neigh = KNeighborsClassifier(n_neighbors = k)
        # Fit model on train data
        neigh.fit(features[train_ind], labels[train_ind])

        # # Compare predicted vs real arrays for debug
        # print("PREDICT")
        # print(gnbclf.predict(features[test_ind]))
        # print("REAL")
        # print(labels[test_ind])
        
        predicts = neigh.predict(features[test_ind])
        true_neg = 0
        false_neg = 0
        true_pos = 0
        false_pos = 0
        
        # Checking for true/false negative and true/false positive predictions
        for i in range(predicts.shape[0]):
            if labels[test_ind][i] == 0:
                if predicts[i] == 0 :
                    true_neg += 1
                elif predicts[i] == 1:
                    false_pos += 1
            elif labels[test_ind][i] == 1:
                if predicts[i] == 0:
                    false_neg += 1
                elif predicts[i] == 1:
                    true_pos += 1
        
        # Calculating Fold sensitivity and specificity
        fsens = true_pos / (true_pos + false_neg)
        fspec = true_neg / (true_neg + false_pos)
        sens = np.append(sens, fsens)
        spec = np.append(spec, fspec)
        accuracy = np.append(accuracy, np.mean(neigh.predict(features[test_ind]) == labels[test_ind]))

    # Printing Accuracy, Sensitivity, Specificity per fold and their mean values
    print("\nAccuracy per fold")
    print(accuracy)
    mean_acc = np.mean(accuracy, axis=0)
    print("\nMean Accuracy")
    print(mean_acc)

    print("\nSensitivity per fold")
    print(sens)
    print("\nMean sensitivity")
    mean_sens = np.mean(sens,axis=0)
    print(mean_sens)

    print("\nSpecificity per fold")
    print(spec)
    print("\nMean Specificity")
    mean_spec = np.mean(spec,axis=0)
    print(mean_spec)

    # Calculating Geometric-Mean per fold and Mean Geometric-Mean
    geo_mean = np.sqrt(sens * spec)
    mean_geo_mean = np.mean(geo_mean, axis=0)
    print("\nGeometric Mean per fold")
    print(geo_mean)
    print("\nMean Geometric Mean")
    print(mean_geo_mean)

    avg_sens = np.append(avg_sens, np.array([mean_sens]), axis=0)
    avg_spec = np.append(avg_spec, np.array([mean_spec]), axis=0)
    avg_geo = np.append(avg_geo, np.array([mean_geo_mean]), axis=0)
    avg_acc = np.append(avg_acc, np.array([mean_acc]), axis=0)

print("\nBEST VALUES")

max_geo_idx = np.argmax(avg_geo, axis=0)
print("\nBest Geometric Mean for K = %d" %(max_geo_idx+3))
print(avg_geo[max_geo_idx])
print("Corresponding Sensitivity")
print(avg_sens[max_geo_idx])
print("Corresponding Specificity")
print(avg_spec[max_geo_idx])
print("Corresponding Accuracy")
print(avg_acc[max_geo_idx])

max_sens_idx = np.argmax(avg_sens, axis=0)
print("\nBest Sensitivity for K = %d" %(max_sens_idx+3))
print(avg_sens[max_sens_idx])
print("Corresponding Specificity")
print(avg_spec[max_sens_idx])
print("Corresponding Geometric Mean")
print(avg_geo[max_sens_idx])
print("Corresponding Accuracy")
print(avg_acc[max_sens_idx])

max_spec_idx = np.argmax(avg_spec, axis=0)
print("\nBest Specificity for K = %d" %(max_spec_idx+3))
print(avg_spec[max_spec_idx])
print("Corresponding Sensitivity")
print(avg_sens[max_spec_idx])
print("Corresponding Geometric Mean")
print(avg_geo[max_spec_idx])
print("Corresponding Accuracy")
print(avg_acc[max_spec_idx])