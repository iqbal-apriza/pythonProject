import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data_train/StressLevelDataset.csv")
df_array = df.to_numpy()

prior_stress = [
    (len(df[df["stress_level"] == 0]) / len(df)) * 10,
    (len(df[df["stress_level"] == 1]) / len(df)) * 10,
    (len(df[df["stress_level"] == 2]) / len(df)) * 10
]

class0 = df[df["stress_level"] == 0]
class1 = df[df["stress_level"] == 1]
class2 = df[df["stress_level"] == 2]

mean_global = df.drop(columns=["stress_level"]).mean().to_numpy()
mean_class0 = class0.drop(columns=["stress_level"]).mean().to_numpy()
mean_class1 = class1.drop(columns=["stress_level"]).mean().to_numpy()
mean_class2 = class2.drop(columns=["stress_level"]).mean().to_numpy()

mean_corrected_0 = np.empty([len(class0), 20])
for i in range(len(class0)):
    for j in range(20):
        mean_corrected_0[i][j] = class0.to_numpy()[i][j] - mean_global[j]

mean_corrected_1 = np.empty([len(class1), 20])
for i in range(len(class1)):
    for j in range(20):
        mean_corrected_1[i][j] = class1.to_numpy()[i][j] - mean_global[j]

mean_corrected_2 = np.empty([len(class2), 20])
for i in range(len(class2)):
    for j in range(20):
        mean_corrected_2[i][j] = class2.to_numpy()[i][j] - mean_global[j]

# cov_class0 = (np.dot(np.matrix_transpose(mean_corrected_0), mean_corrected_0)) / len(mean_corrected_0)
# cov_class1 = (np.dot(np.matrix_transpose(mean_corrected_1), mean_corrected_1)) / len(mean_corrected_1)
# cov_class2 = (np.dot(np.matrix_transpose(mean_corrected_2), mean_corrected_2)) / len(mean_corrected_2)

cov_class0 = np.dot(mean_corrected_0.T, mean_corrected_0) / len(mean_corrected_0)
cov_class1 = np.dot(mean_corrected_1.T, mean_corrected_1) / len(mean_corrected_1)
cov_class2 = np.dot(mean_corrected_2.T, mean_corrected_2) / len(mean_corrected_2)


cov_matrix = np.empty([20, 20])
for i in range(20):
    for j in range(20):
        cov_matrix[i][j] = (cov_class0[i][j] * prior_stress[0]) + (cov_class1[i][j] * prior_stress[1]) + (cov_class2[i][j] * prior_stress[2])

inv_cov = np.linalg.inv(cov_matrix)

term1 = np.zeros([3])
term1[0] = 0.5 * np.dot(mean_class0.T, np.dot(inv_cov, mean_class0))
term1[1] = 0.5 * np.dot(mean_class1.T, np.dot(inv_cov, mean_class1))
term1[2] = 0.5 * np.dot(mean_class2.T, np.dot(inv_cov, mean_class2))

ln = np.zeros([3])
ln[0] = np.log(prior_stress[0]) / np.log(np.e)
ln[1] = np.log(prior_stress[1]) / np.log(np.e)
ln[2] = np.log(prior_stress[2]) / np.log(np.e)

disc = np.zeros([len(df), 3])
for i in range(len(df)):
    disc[i][0] = np.dot(df.drop(columns=['stress_level']).to_numpy()[i], np.dot(inv_cov, mean_class0)) - term1[0] + ln[0]
    disc[i][1] = np.dot(df.drop(columns=['stress_level']).to_numpy()[i], np.dot(inv_cov, mean_class1)) - term1[1] + ln[1]
    disc[i][2] = np.dot(df.drop(columns=['stress_level']).to_numpy()[i], np.dot(inv_cov, mean_class2)) - term1[2] + ln[2]

# for i in range(len(df)):
#     x = df.drop(columns=['stress_level']).to_numpy()[i].ravel()
#     disc[i][0] = np.dot(x, np.dot(inv_cov, mean_class0)) - term1[0] + ln[0]
#     disc[i][1] = np.dot(x, np.dot(inv_cov, mean_class1)) - term1[1] + ln[1]
#     disc[i][2] = np.dot(x, np.dot(inv_cov, mean_class2)) - term1[2] + ln[2]

# disc[i][0] = np.dot(mean_class0, inv_cov) * df.drop(columns=['stress_level']).to_numpy()[i] - term1[0] + ln[0]
# disc[i][1] = np.dot(mean_class1, inv_cov) * df.drop(columns=['stress_level']).to_numpy()[i] - term1[1] + ln[1]
# disc[i][2] = np.dot(mean_class2, inv_cov) * df.drop(columns=['stress_level']).to_numpy()[i] - term1[2] + ln[2]

# disc_f0 = np.zeros([len(df), 20])
# disc_f1 = np.zeros([len(df), 20])
# disc_f2 = np.zeros([len(df), 20])

# for i in range(len(df)):
#     disc_f0[i] =
#     disc_f1[i] = np.dot(mean_class1, inv_cov) * df.drop(columns=['stress_level']).to_numpy()[i] - test[1] + ln[1]
#     disc_f2[i] = np.dot(mean_class2, inv_cov) * df.drop(columns=['stress_level']).to_numpy()[i] - test[2] + ln[2]

# print(np.dot(mean_class0, inv_cov) * df.drop(columns=['stress_level']).to_numpy()[i] - term1[0] + ln[0])

print(disc)
predicted_result = np.argmax(disc, axis=1)
actual_result = df['stress_level']

result = pd.DataFrame({
    "Predicted" : predicted_result,
    "Actual" : actual_result,
    "Error" : predicted_result - actual_result
})

print(result)

how_much = len(df)
conf_matrix = np.empty([3, 3], dtype=int)
for i in range(3):
    for j in range(3):
        conf_matrix[i][j] = len(result[(result["Actual"] == i) & (result["Predicted"] == j)])

df_conf_matrix = pd.DataFrame(conf_matrix)

accuracy = ((conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2]) / how_much) * 100

precision_ = np.empty([3])
for i in range(3):
    precision_[i] = (conf_matrix[i][i] / sum(conf_matrix[i])) * 100
precision_avg = np.average(precision_)

recall = np.empty([3])
for i in range(3):
    recall[i] = (conf_matrix[i][i] / sum(conf_matrix[:, i])) * 100
recall_avg = np.average(recall)

f1_score = 2 * ((recall_avg * precision_avg) / (recall_avg + precision_avg))

print(result)
print("")
print(df_conf_matrix)
print("")
print(f"Accuracy = {accuracy}")
print(f"Precision = {np.round(precision_, 3)}")
print(f"Avg Precision = {precision_avg}")
print(f"Recall = {recall}")
print(f"Avg Recall = {recall_avg}")
print(f"F1 Score = {f1_score}")



# plt.scatter(np.max(disc_f0, axis=0), 0)
# plt.show()
# print(np.matrix_transpose(mean_class0))
# print(pd.DataFrame(cov_matrix))
# print("")
# print(np.linalg.inv(cov_matrix))
# print("")
# print(pd.DataFrame(mean_corrected_2))
# print("")
# print(cov_class2)