from tugas_pengpol.big_project.naive_bayes_log.main import *

# Test Data
test_data_array = sample_data.drop(columns=["stress_level"]).to_numpy()
actual_result = sample_data["stress_level"]

# Calculate Log Probabilities
log_probability = np.zeros((len(test_data_array), len(stress_levels)))

for j, data_point in enumerate(test_data_array):
    for i in stress_levels:
        log_probability[j][i] = (
            np.log(prior_stress[i]) +
            np.log(anxiety_prob[data_point[0]][i]) +
            np.log(self_est_prob[data_point[1]][i]) +
            np.log(mental_hlt_prob[data_point[2]][i]) +
            np.log(depress_prob[data_point[3]][i]) +
            np.log(headache_prob[data_point[4]][i]) +
            np.log(blood_prs_prob[data_point[5] - 1][i]) +
            np.log(sleep_qly_prob[data_point[6]][i]) +
            np.log(breath_prob[data_point[7]][i]) +
            np.log(noise_lvl_prob[data_point[8]][i]) +
            np.log(living_cond_prob[data_point[9]][i]) +
            np.log(safety_prob[data_point[10]][i]) +
            np.log(basic_need_prob[data_point[11]][i]) +
            np.log(academic_prob[data_point[12]][i]) +
            np.log(study_load_prob[data_point[13]][i]) +
            np.log(ts_relation_prob[data_point[14]][i]) +
            np.log(future_career_prob[data_point[15]][i]) +
            np.log(social_prob[data_point[16]][i]) +
            np.log(peer_prs_prob[data_point[17]][i]) +
            np.log(extracurricular_prob[data_point[18]][i]) +
            np.log(bullying_prob[data_point[19]][i])
        )

# Normalize using Softmax
log_probability -= np.max(log_probability, axis=1, keepdims=True)  # Avoid overflow
probability2 = np.exp(log_probability)
probability2 /= np.sum(probability2, axis=1, keepdims=True)

# Predictions
predicted_result = np.argmax(probability2, axis=1)
result = pd.DataFrame({
    "Predicted": predicted_result,
    "Actual": actual_result.values,
    "Error": predicted_result - actual_result.values
})

# Confusion Matrix and Metrics
conf_matrix = np.zeros((len(stress_levels), len(stress_levels)), dtype=int)
for i in range(len(stress_levels)):
    for j in range(len(stress_levels)):
        conf_matrix[i][j] = len(result[(result["Actual"] == i) & (result["Predicted"] == j)])

accuracy = np.trace(conf_matrix) / len(test_data_array) * 100
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1) * 100
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0) * 100
f1_score = 2 * (np.mean(precision) * np.mean(recall)) / (np.mean(precision) + np.mean(recall))

# Results
print("\nHasil Training:")
print(result)
print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix))
print(f"\nAccuracy = {accuracy:.2f}%")
print(f"Precision = {precision}")
print(f"Avg Precision = {np.mean(precision):.2f}%")
print(f"Recall = {recall}")
print(f"Avg Recall = {np.mean(recall):.2f}%")
print(f"F1 Score = {f1_score:.2f}")