import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
# 4: segment type, 5:typ,
from CONSTANTS import *
#from feature_plot import plot_feature_imp

dataset = pd.read_csv(DATASET_PATH, delimiter=DELIMITER, usecols=[4, 5, 13, 17, 11, 18, 14])
#print(dataset.head())
segment = dataset.iloc[:, 0].astype('category')
typ = dataset.iloc[:, 1].astype('category')
encoded_segment = segment.cat.codes
encoded_typ = typ.cat.codes
intensity_metric_data = dataset.iloc[:, 4]
intensity_feature = []

#preprocess_intensity_metric(intensity_metric_data)

for (idx, items) in intensity_metric_data.iteritems():
    items = items.replace("'", "").replace('g', '').replace('/', '').translate(str.maketrans({"'": None})).split()
    conv_list = [float(i) for i in items]
    items = tuple(conv_list)
    intensity_feature.append(items)
a = dataset.iloc[:, 2]
b = dataset.iloc[:, 5]
c = dataset.iloc[:, 6]
a = pd.to_datetime(a, format='%d.%m.%Y %H:%M:%S')
b = pd.to_datetime(b, format='%d.%m.%Y %H:%M:%S')
c = pd.to_datetime(c, format='%d.%m.%Y %H:%M:%S')
intensity_feature = pd.DataFrame(intensity_feature, columns=['Intensity_metric_1', 'Intensity_metric_2'])
shock_diff = pd.to_timedelta(a - b)
runnin_diff = pd.to_timedelta(c - b)
shock_diff = shock_diff.dt.total_seconds()/(60*60)  # Converting to hours
runnin_diff = runnin_diff.dt.total_seconds()/(60*60)  # Converting to hours

data = pd.DataFrame([encoded_segment, shock_diff, runnin_diff, encoded_typ, dataset.iloc[:, 3]],
                    index = ['Segment', 'Time_difference_of_Shock', 'Difference_in_usage_Interval', 'Type', 'Shock_level']).transpose()
total_data = pd.concat([intensity_feature, data], axis=1)
total_data = total_data[total_data.iloc[:, 3].notna()]
total_data = total_data[total_data.iloc[:, 4].notna()]
#total_data = shuffle(total_data)
X_train, X_test, y_train, y_test = train_test_split(total_data.iloc[:, :-1], total_data.iloc[:, -1], test_size=0.33)
print(total_data.head())
# TODO:
# 1. Write a clean and structured code - train.py, evaluation.py
# 2. Start visualizing the Confusion matrix
# 3. Write the README file
# 4. Create a git repo and push it
# 5. Visualization for feature importance - bar graph
# 6. Visualize Segment wise shock levels
# 7. Flags for multiple models and write in readme how to activate them
# 8. Future scopes - predicting the shock timing, shock alert
# 9. Why current approach: we can decide if this segment vehicle runs for so and so hours, what will be the level of shock (anticipate)
# 10. Write


#dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
dtree_model = RandomForestClassifier()
dtree_model.fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

feature_importance = dtree_model.feature_importances_
feature_importance = feature_importance.round(decimals=2)
#print(dtree_model.feature_importances_)
feature_names = np.asarray(list(total_data.columns.values)).reshape(-1, 1)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
#disp = plot_confusion_matrix(dtree_model, X_test, y_test,
#                                 display_labels=['shock_1', 'shock_2', 'shock_3'],
#                                 cmap=plt.cm.Blues)
#plt.show()
#print(cm)
#print('----------')

# How's our accuracy?
accuracy = accuracy_score(y_test, dtree_predictions)
precision_recall_fscore = precision_recall_fscore_support(y_test, dtree_predictions, average='weighted')

print('Accuracy: {}, Precision: {:.2f}, Recall: {:.2f}'.format(accuracy, precision_recall_fscore[0], precision_recall_fscore[1]))