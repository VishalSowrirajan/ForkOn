def preprocess_intensity_metric(intensity_data):
    intensity_feature = []
    for (idx, items) in intensity_data.iteritems():
        items = items.replace("'", "").replace('g', '').replace('/', '').translate(str.maketrans({"'": None})).split()
        conv_list = [float(i) for i in items]
        intensity_feature.append(tuple(conv_list))
    return pd.DataFrame(intensity_feature)

import pandas as pd
from CONSTANTS import *


def convert_to_numeric_features(feature):
    return feature.cat.codes

class Preprocessor:

    def __init__(self, dataset):
        self.dataset_path = dataset

    def parse_dataset(self):
        dataset = pd.read_csv(self.dataset_path, delimiter=DELIMITER, usecols=[4, 5, 13, 17, 11, 18, 14])
        segment = dataset.iloc[:, 0].astype('category')
        typ = dataset.iloc[:, 1].astype('category')
        intensity_metric_data = dataset.iloc[:, 4]

        encoded_segment = convert_to_numeric_features(segment)
        encoded_typ = convert_to_numeric_features(typ)
        preprocess_intensity_metric(intensity_metric_data)

        a = dataset.iloc[:, 2]
        b = dataset.iloc[:, 5]
        c = dataset.iloc[:, 6]
        a = pd.to_datetime(a, format='%d.%m.%Y %H:%M:%S')
        b = pd.to_datetime(b, format='%d.%m.%Y %H:%M:%S')
        c = pd.to_datetime(c, format='%d.%m.%Y %H:%M:%S')

    def extract_feature(self):
