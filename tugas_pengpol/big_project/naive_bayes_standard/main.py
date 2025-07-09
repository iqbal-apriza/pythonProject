import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data_train/StressLevelDataset.csv")

df, sample_data = train_test_split(
    data, test_size=0.2, random_state=42
)

# Prior Probability
prior_stress = [
    (len(df[df["stress_level"] == 0]) / len(df)) * 10,
    (len(df[df["stress_level"] == 1]) / len(df)) * 10,
    (len(df[df["stress_level"] == 2]) / len(df)) * 10
]

# Conditional Probability
anxiety_prob = np.empty([df['anxiety_level'].max()+1, 3])
for i in range(df['anxiety_level'].max()+1):
    for j in range(3):
        anxiety_prob[i][j] = (len(df[(df['anxiety_level'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

self_est_prob = np.empty([df['self_esteem'].max()+1, 3])
for i in range(df['self_esteem'].max()+1):
    for j in range(3):
        self_est_prob[i][j] = (len(df[(df['self_esteem'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

mental_hlt_prob = np.empty([df['mental_health_history'].max()+1, 3])
for i in range(df['mental_health_history'].max()+1):
    for j in range(3):
        mental_hlt_prob[i][j] = (len(df[(df['mental_health_history'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

depress_prob = np.empty([df['depression'].max()+1, 3])
for i in range(df['depression'].max()+1):
    for j in range(3):
        depress_prob[i][j] = (len(df[(df['depression'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

headache_prob = np.empty([df['headache'].max()+1, 3])
for i in range(df['headache'].max()+1):
    for j in range(3):
        headache_prob[i][j] = (len(df[(df['headache'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

blood_prs_prob = np.empty([df['blood_pressure'].max(), 3])
for i in range(df['blood_pressure'].max()):
    for j in range(3):
        blood_prs_prob[i][j] = (len(df[(df['blood_pressure'] == i+1) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

sleep_qly_prob = np.empty([df['sleep_quality'].max()+1, 3])
for i in range(df['sleep_quality'].max()+1):
    for j in range(3):
        sleep_qly_prob[i][j] = (len(df[(df['sleep_quality'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

breath_prob = np.empty([df['breathing_problem'].max()+1, 3])
for i in range(df['breathing_problem'].max()+1):
    for j in range(3):
        breath_prob[i][j] = (len(df[(df['breathing_problem'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

noise_lvl_prob = np.empty([df['noise_level'].max()+1, 3])
for i in range(df['noise_level'].max()+1):
    for j in range(3):
        noise_lvl_prob[i][j] = (len(df[(df['noise_level'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

living_cond_prob = np.empty([df['living_conditions'].max()+1, 3])
for i in range(df['living_conditions'].max()+1):
    for j in range(3):
        living_cond_prob[i][j] = (len(df[(df['living_conditions'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

safety_prob = np.empty([df['safety'].max()+1, 3])
for i in range(df['safety'].max()+1):
    for j in range(3):
        safety_prob[i][j] = (len(df[(df['safety'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

basic_need_prob = np.empty([df['basic_needs'].max()+1, 3])
for i in range(df['basic_needs'].max()+1):
    for j in range(3):
        basic_need_prob[i][j] = (len(df[(df['basic_needs'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

academic_prob = np.empty([df['academic_performance'].max()+1, 3])
for i in range(df['academic_performance'].max()+1):
    for j in range(3):
        academic_prob[i][j] = (len(df[(df['academic_performance'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

study_load_prob = np.empty([df['study_load'].max()+1, 3])
for i in range(df['study_load'].max()+1):
    for j in range(3):
        study_load_prob[i][j] = (len(df[(df['study_load'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

ts_relation_prob = np.empty([df['teacher_student_relationship'].max()+1, 3])
for i in range(df['teacher_student_relationship'].max()+1):
    for j in range(3):
        ts_relation_prob[i][j] = (len(df[(df['teacher_student_relationship'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

future_career_prob = np.empty([df['future_career_concerns'].max()+1, 3])
for i in range(df['future_career_concerns'].max()+1):
    for j in range(3):
        future_career_prob[i][j] = (len(df[(df['future_career_concerns'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

social_prob = np.empty([df['social_support'].max()+1, 3])
for i in range(df['social_support'].max()+1):
    for j in range(3):
        social_prob[i][j] = (len(df[(df['social_support'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

peer_prs_prob = np.empty([df['peer_pressure'].max()+1, 3])
for i in range(df['peer_pressure'].max()+1):
    for j in range(3):
        peer_prs_prob[i][j] = (len(df[(df['peer_pressure'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

extracurricular_prob = np.empty([df['extracurricular_activities'].max()+1, 3])
for i in range(df['extracurricular_activities'].max()+1):
    for j in range(3):
        extracurricular_prob[i][j] = (len(df[(df['extracurricular_activities'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10

bullying_prob = np.empty([df['bullying'].max()+1, 3])
for i in range(df['bullying'].max()+1):
    for j in range(3):
        bullying_prob[i][j] = (len(df[(df['bullying'] == i) & (df['stress_level'] == j)]) / len(df['stress_level'] == j)) * 10