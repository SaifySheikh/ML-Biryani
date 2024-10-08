
k = 10
length = [10.0, 11.0, 12.0, 7.0, 9.0, 8.0,6.0,15.0,14.0,7.0,10.0,13.0,9.0,5.0,5.0]
weight = [15.0, 6.0 , 14.0 , 9.0 ,14.0,12.0,11.0,10.0,8.0,12.0,6.0,8.0,7.0,8.0,10.0]
cost = [45,37,48,33,38,40,35,50,46,35,36,44,32,30,30]

test_length = 7
test_weight = 8

def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

distances = [(euclidean_distance((test_length, test_weight), (length[i], weight[i])), cost[i]) for i in range(len(length))]


sorted_distances = sorted(distances, key=lambda x: x[0])

k_nearest_neighbors = sorted_distances[:k]

weighted_cost_sum = sum(neighbor[1] / neighbor[0] for neighbor in k_nearest_neighbors)
weighted_distance_sum = sum(1 / neighbor[0] for neighbor in k_nearest_neighbors)
predicted_cost = weighted_cost_sum / weighted_distance_sum



print(f"Predicted cost for (length={test_length}, weight={test_weight}): {predicted_cost:.2f}")

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


iris = datasets.load_iris()


X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Confusion Matrix : ", end="\n\n")
print(confusion_mat, end="\n\n\n\n")
print("\nClassification Report : ", end="\n\n")
print(classification_rep)

new_data = np.array([[5.1, 3.5, 1.4, 0.2]])


predicted_class = knn_classifier.predict(new_data)

predicted_class_label = iris.target_names[predicted_class[0]]


print(f"The predicted class for new_data is: {predicted_class_label}")

