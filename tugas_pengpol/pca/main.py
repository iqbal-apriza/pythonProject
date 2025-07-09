import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Membuka file csv
with open('datasets/student_performance_data.csv', mode='r') as file:
    read_csv = csv.reader(file, delimiter=',')
    data = list(read_csv)

# Ubah data ke dalam bentuk tabel
df = pd.DataFrame(data, columns=['StudentID', 'Gender', 'Age', 'StudyHoursPerWeek', 'AttendanceRate', 'GPA', 'Major', 'PartTimeJob', 'ExtraCurricularActivities'])

# Ubah nilai numerik menjadi bentuk float
df[['Age', 'StudyHoursPerWeek', 'AttendanceRate', 'GPA']] = df[['Age', 'StudyHoursPerWeek', 'AttendanceRate', 'GPA']].apply(pd.to_numeric, errors='coerce')

# Menghapus baris yang bernilai NaN
df.dropna(subset=['Age', 'StudyHoursPerWeek', 'AttendanceRate', 'GPA'], inplace=True)

# Standarisasi nilai
df_standardized = (df[['Age', 'StudyHoursPerWeek', 'AttendanceRate', 'GPA']] - df[['Age', 'StudyHoursPerWeek', 'AttendanceRate', 'GPA']].mean()) / df[['Age', 'StudyHoursPerWeek', 'AttendanceRate', 'GPA']].std()

# Menghitung matriks covariance
cov_matrix = np.cov(df_standardized.T)

# Menghitung eigenvalue dan eigenvector
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Urutkan eigenvalue dan eigenvector
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Memilih 2 eigenvector pertama sebagai PC1 dan PC2
n_components = 2
principal_components = sorted_eigenvectors[:, :n_components]

# Transformasi data
df_pca = np.dot(df_standardized, principal_components)

# Membuat DataFrame untuk menampilkan hasil PCA
df_pca = pd.DataFrame(data=df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])])

# Menambah kolom StudentID
df_pca.index = df['StudentID']

# Menampilkan hasil PCA
print(df_pca)

# Mapping warna untuk jenis kelamin untuk plot grafik
colors = {'Male': 'blue', 'Female': 'red'}
plt.figure(figsize=(10, 7))

# Plot nilai-nilai sesuai dengan jenis kelamin
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df['Gender'].map(colors), marker='o')

# Menandai point dengan StudentID
for i, student_id in enumerate(df_pca.index):
    plt.annotate(student_id, (df_pca['PC1'][i], df_pca['PC2'][i]), fontsize=9, ha='right')

# Menentukan label dari aksis koordinat
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Judul plot
plt.title('PCA Result - 2D Scatter Plot (Colored by Gender)')

# Menampilkan plot grafik
plt.show()
