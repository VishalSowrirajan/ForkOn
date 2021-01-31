import pickle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, plot_confusion_matrix, confusion_matrix
import numpy as np
from sklearn.metrics import mean_squared_error



class BaggingRegressorModel:

    def __init__(self, model_name):
        self.model_name = model_name

    def train_model(self, X_train, y_train, save_model_path):
        self.model_name.fit(X_train, y_train)
        pickle.dump(self.model_name, open(save_model_path, 'wb'))

    def test_model(self, X_test, y_test, filename):
        loaded_model = pickle.load(open(filename, 'rb'))
        classifications = loaded_model.predict(X_test)
        mse_error = np.sqrt(mean_squared_error(y_test, classifications))
        model_score = loaded_model.score(X_test, y_test)
        return classifications, model_score, mse_error
