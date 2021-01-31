from sklearn import tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from CONSTANTS import *
from classFiles.model import BaggingRegressorModel
from data.Preprocessing import Preprocessor
from utils import get_dataset_path
from sklearn.linear_model import LinearRegression

dataset_name = get_dataset_path(DATASET_PATH)
checkpoint_path = get_dataset_path(SAVE_MODEL_PATH)

data_parser = Preprocessor(dataset=dataset_name)
features = data_parser.parse_dataset()

X_train, X_test, y_train, y_test = train_test_split(features.iloc[:, :-1], features.iloc[:, -1], test_size=0.33)

# Model selection - we choose RANDOM FOREST as our classifier
model = BaggingRegressorModel(model_name=LinearRegression())

# Train the model and Save the weights
model.train_model(X_train, y_train, checkpoint_path)

# Test the model on Test data
clasification_results, model_score, mse_error = model.test_model(X_test, y_test, checkpoint_path)

print('Accuracy: {:.2f}%, MSE_Error: {:.2f}'.format(model_score*100, mse_error))