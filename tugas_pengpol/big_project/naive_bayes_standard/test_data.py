from main import *
from tugas_pengpol.big_project.generate_data import *

testing = pd.read_csv("../data_test/test_data2.csv")

# test_data_array = random_data_array
test_data_array = testing.to_numpy()

probability = np.empty([20, 3])
for j in range(20):
    for i in range(3):
        probability[j][i] = prior_stress[i] * anxiety_prob[test_data_array[j][0]][i] * self_est_prob[test_data_array[j][1]][i] * mental_hlt_prob[test_data_array[j][2]][i] * depress_prob[test_data_array[j][3]][i] * headache_prob[test_data_array[j][4]][i] * blood_prs_prob[test_data_array[j][5]-1][i] * sleep_qly_prob[test_data_array[j][6]][i] * breath_prob[test_data_array[j][7]][i] * noise_lvl_prob[test_data_array[j][8]][i] * living_cond_prob[test_data_array[j][9]][i] * safety_prob[test_data_array[j][10]][i] * basic_need_prob[test_data_array[j][11]][i] * academic_prob[test_data_array[j][12]][i] * study_load_prob[test_data_array[j][13]][i] * ts_relation_prob[test_data_array[j][14]][i] * future_career_prob[test_data_array[j][15]][i] * social_prob[test_data_array[j][16]][i] * peer_prs_prob[test_data_array[j][17]][i] * extracurricular_prob[test_data_array[j][18]][i] * bullying_prob[test_data_array[j][19]][i]

probability2 = np.empty([20, 3])
for j in range(20):
    for i in range(3):
        probability2[j][i] = round(probability[j][i] / sum(probability[j]), 4)

result = pd.DataFrame({
    "Stress L1" : probability2[:, 0],
    "Stress L2" : probability2[:, 1],
    "Stress L3" : probability2[:, 2],
    "Kategori" : np.argmax(probability2, axis=1) + 1
})
print(result.fillna("Undefined"))
