import pickle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, plot_confusion_matrix, confusion_matrix


class RandomForestClassifierModel:

    def __init__(self, model_name):
        self.model_name = model_name

    def train_model(self, X_train, y_train, save_model_path):
        self.model_name.fit(X_train, y_train)
        pickle.dump(self.model_name, open(save_model_path, 'wb'))

    def test_model(self, X_test, filename):
        loaded_model = pickle.load(open(filename, 'rb'))
        classifications = loaded_model.predict(X_test)
        return classifications

    def calculate_model_performance(self, y_test, prediction_results):
        cm = confusion_matrix(y_test, prediction_results)
        accuracy = accuracy_score(y_test, prediction_results)
        precision_recall_fscore = precision_recall_fscore_support(y_test, prediction_results, average='weighted')
        precision = precision_recall_fscore[0]
        recall = precision_recall_fscore[1]
        return cm, accuracy, precision, recall
