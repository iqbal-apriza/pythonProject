import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("../data_train/StressLevelDataset.csv")

df, sample_data = train_test_split(data, test_size=0.2, random_state=42)

# Laplace smoothing factor
alpha = 1

# Compute Prior Probabilities
stress_levels = [0, 1, 2]
prior_stress = [(len(df[df["stress_level"] == level]) + alpha) / (len(df) + len(stress_levels) * alpha)
                for level in stress_levels]

# Compute Conditional Probabilities with Laplace smoothing
def compute_conditional_probabilities(column_name, levels, target_levels):
    max_val = df[column_name].max() + 1
    probs = np.empty((max_val, len(target_levels)))
    for i in range(max_val):
        for j in target_levels:
            probs[i][j] = ((len(df[(df[column_name] == i) & (df["stress_level"] == j)]) + alpha) /
                           (len(df[df["stress_level"] == j]) + alpha * max_val))
    return probs

anxiety_prob = compute_conditional_probabilities('anxiety_level', df['anxiety_level'], stress_levels)
self_est_prob = compute_conditional_probabilities('self_esteem', df['self_esteem'], stress_levels)
mental_hlt_prob = compute_conditional_probabilities('mental_health_history', df['mental_health_history'], stress_levels)
depress_prob = compute_conditional_probabilities('depression', df['depression'], stress_levels)
headache_prob = compute_conditional_probabilities('headache', df['headache'], stress_levels)
blood_prs_prob = compute_conditional_probabilities('blood_pressure', df['blood_pressure'], stress_levels)
sleep_qly_prob = compute_conditional_probabilities('sleep_quality', df['sleep_quality'], stress_levels)
breath_prob = compute_conditional_probabilities('breathing_problem', df['breathing_problem'], stress_levels)
noise_lvl_prob = compute_conditional_probabilities('noise_level', df['noise_level'], stress_levels)
living_cond_prob = compute_conditional_probabilities('living_conditions', df['living_conditions'], stress_levels)
safety_prob = compute_conditional_probabilities('safety', df['safety'], stress_levels)
basic_need_prob = compute_conditional_probabilities('basic_needs', df['basic_needs'], stress_levels)
academic_prob = compute_conditional_probabilities('academic_performance', df['academic_performance'], stress_levels)
study_load_prob = compute_conditional_probabilities('study_load', df['study_load'], stress_levels)
ts_relation_prob = compute_conditional_probabilities('teacher_student_relationship', df['teacher_student_relationship'], stress_levels)
future_career_prob = compute_conditional_probabilities('future_career_concerns', df['future_career_concerns'], stress_levels)
social_prob = compute_conditional_probabilities('social_support', df['social_support'], stress_levels)
peer_prs_prob = compute_conditional_probabilities('peer_pressure', df['peer_pressure'], stress_levels)
extracurricular_prob = compute_conditional_probabilities('extracurricular_activities', df['extracurricular_activities'], stress_levels)
bullying_prob = compute_conditional_probabilities('bullying', df['bullying'], stress_levels)