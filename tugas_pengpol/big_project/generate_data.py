import numpy as np
import pandas as pd

random_data = {
    'anxiety_level' : np.random.randint(0, 21, size=20),
    'self_esteem' : np.random.randint(0, 30, size=20),
    'mental_health_history' : np.random.randint(0, 1, size=20),
    'depression' : np.random.randint(0, 27, size=20),
    'headache' : np.random.randint(0, 5, size=20),
    'blood_pressure' : np.random.randint(1, 3, size=20),
    'sleep_quality' : np.random.randint(0, 5, size=20),
    'breathing_problem' : np.random.randint(0, 5, size=20),
    'noise_level' : np.random.randint(0, 5, size=20),
    'living_conditions' : np.random.randint(0, 5, size=20),
    'safety' : np.random.randint(0, 5, size=20),
    'basic_needs' : np.random.randint(0, 5, size=20),
    'academic_performance' : np.random.randint(0, 5, size=20),
    'study_load' : np.random.randint(0, 5, size=20),
    'teacher_student_relationship' : np.random.randint(0, 5, size=20),
    'future_career_concerns' : np.random.randint(0, 5, size=20),
    'social_support' : np.random.randint(0, 3, size=20),
    'peer_pressure' : np.random.randint(0, 5, size=20),
    'extracurricular_activities' : np.random.randint(0, 5, size=20),
    'bullying' : np.random.randint(0, 5, size=20)
}

random_data_table = pd.DataFrame(random_data)
random_data_array = random_data_table.to_numpy()