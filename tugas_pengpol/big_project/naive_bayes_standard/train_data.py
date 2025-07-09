from main import *

test_data_array = sample_data.drop(columns=["stress_level"]).to_numpy()
actual_result = sample_data["stress_level"]

probability = np.empty([len(test_data_array), 3])
for j in range(len(test_data_array)):
    for i in range(3):
        probability[j][i] = prior_stress[i] * anxiety_prob[test_data_array[j][0]][i] * self_est_prob[test_data_array[j][1]][i] * mental_hlt_prob[test_data_array[j][2]][i] * depress_prob[test_data_array[j][3]][i] * headache_prob[test_data_array[j][4]][i] * blood_prs_prob[test_data_array[j][5]-1][i] * sleep_qly_prob[test_data_array[j][6]][i] * breath_prob[test_data_array[j][7]][i] * noise_lvl_prob[test_data_array[j][8]][i] * living_cond_prob[test_data_array[j][9]][i] * safety_prob[test_data_array[j][10]][i] * basic_need_prob[test_data_array[j][11]][i] * academic_prob[test_data_array[j][12]][i] * study_load_prob[test_data_array[j][13]][i] * ts_relation_prob[test_data_array[j][14]][i] * future_career_prob[test_data_array[j][15]][i] * social_prob[test_data_array[j][16]][i] * peer_prs_prob[test_data_array[j][17]][i] * extracurricular_prob[test_data_array[j][18]][i] * bullying_prob[test_data_array[j][19]][i]

probability2 = np.empty([len(test_data_array), 3])
for j in range(len(test_data_array)):
    for i in range(3):
        probability2[j][i] = round(probability[j][i] / sum(probability[j]), 4)

predicted_result = np.argmax(probability2, axis=1)
result = pd.DataFrame({
    "Predicted" : predicted_result,
    "Actual" : actual_result,
    "Error" : predicted_result - actual_result
})

conv_matrix = np.empty([3, 3], dtype=int)
for i in range(3):
    for j in range(3):
        conv_matrix[i][j] = len(result[(result["Actual"] == i) & (result["Predicted"] == j)])

df_conv_matrix = pd.DataFrame(conv_matrix)

accuracy = ((conv_matrix[0][0] + conv_matrix[1][1] + conv_matrix[2][2]) / len(test_data_array)) * 100

precision_ = np.empty([3])
for i in range(3):
    precision_[i] = (conv_matrix[i][i] / sum(conv_matrix[i])) * 100
precision_avg = np.average(precision_)

recall = np.empty([3])
for i in range(3):
    recall[i] = (conv_matrix[i][i] / sum(conv_matrix[:, i])) * 100
recall_avg = np.average(recall)

f1_score = 2 * ((recall_avg * precision_avg) / (recall_avg + precision_avg))

print("Hasil Training:")
print(result)
print("")
print("Confusion Matrix")
print(df_conv_matrix)
print("")
print(f"Accuracy = {accuracy:0,.3f}")
print(f"Precision = {np.round(precision_, 3)}")
print(f"Avg Precision = {precision_avg:0,.2f}")
print(f"Recall = {np.round(recall, 3)}")
print(f"Avg Recall = {recall_avg:0,.3f}")
print(f"F1 Score = {f1_score:0,.3f}")