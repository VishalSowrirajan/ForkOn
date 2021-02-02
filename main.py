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

# Model selection - we choose RANDOM FOREST as our classifier
model = RandomForestClassifierModel(model_name=RandomForestClassifier())

# Train the model and Save the weights
model.train_model(X_train, y_train, checkpoint_path)

# Test the model on Test data
clasification_results = model.test_model(X_test, checkpoint_path)

# Calculate the performance metric
cm, accuracy, precision, recall = model.calculate_model_performance(y_test, clasification_results)

print('Accuracy: {}, Precision: {:.2f}, Recall: {:.2f}'.format(accuracy, precision, recall))