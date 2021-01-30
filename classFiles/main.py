from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from CONSTANTS import *
from classFiles.model import RandomForestClassifierModel
from data.Preprocessing import Preprocessor
from utils import get_dataset_path

dataset_name = get_dataset_path(DATASET_PATH)
checkpoint_path = get_dataset_path(SAVE_MODEL_PATH)

data_parser = Preprocessor(dataset=dataset_name)
features = data_parser.parse_dataset()

X_train, X_test, y_train, y_test = train_test_split(features.iloc[:, :-1], features.iloc[:, -1], test_size=0.33)
# TODO:
# 1. Write a clean and structured code - train.py, evaluation.py - Done
# 2. Start visualizing the Confusion matrix - Done
# 3. Write the README file
# 4. Create a git repo and push it  - Done
# 5. Visualization for feature importance - bar graph - Done
# 6. Visualize Segment wise shock levels
# 7. Flags for multiple models and write in readme how to activate them
# 8. Future scopes - predicting the shock timing, shock alert
# 9. Why current approach: we can decide if this segment vehicle runs for so and so hours, what will be the level of shock (anticipate)

model = RandomForestClassifierModel(model_name=RandomForestClassifier())

# Train the model and Save the weights
model.train_model(X_train, y_train, checkpoint_path)

# Test the model on Test data
clasification_results = model.test_model(X_test, checkpoint_path)

# Calculate the performance metric
cm, accuracy, precision, recall = model.calculate_model_performance(y_test, clasification_results)

print('Accuracy: {}, Precision: {:.2f}, Recall: {:.2f}'.format(accuracy, precision, recall))