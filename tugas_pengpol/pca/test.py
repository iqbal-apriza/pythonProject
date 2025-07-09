import numpy as np

data = [
    ['class', 'feature1', 'feature2', 'feature3', 'feature4'],
    ['class1', '10', '3', '2', '12'],
    ['class2', '2', '10', '13', '2'],
    ['class3', '5', '8', '14', '10'],
    ['class4', '12', '10', '4', '3'],
    ['class5', '1', '4', '14', '12']
]

features = {}

for i, feature in enumerate(data[0][1:], start=1):
    features[feature] = []

for row in data[1:]:
    for i, feature in enumerate(data[0][1:], start=1):
        features[feature].append(int(row[i]))

averages = {}

for feature, values in features.items():
    averages[feature] = sum(values) / len(values)

zeromean = {}

for feature in features.items():
    zeromean[feature[0]] = []

for average in averages.items():
    for values in features[average[0]]:
        zeromean[average[0]].append(values - average[1])

for zmean in zeromean.items():
    print(zmean)