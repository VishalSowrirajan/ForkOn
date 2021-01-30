import pandas as pd
from CONSTANTS import *
from utils import *


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
        intensity = preprocess_intensity_metric(intensity_metric_data)

        time_of_shock = pd.to_datetime(dataset.iloc[:, 2], format='%d.%m.%Y %H:%M:%S')
        begin_time = pd.to_datetime(dataset.iloc[:, 5], format='%d.%m.%Y %H:%M:%S')
        end_time = pd.to_datetime(dataset.iloc[:, 6], format='%d.%m.%Y %H:%M:%S')

        shock_time_interval = calc_time_difference(time_of_shock, begin_time)
        op_time_interval = calc_time_difference(end_time, begin_time)

        shock_interval_in_hours = convert_to_hours(shock_time_interval)
        op_interval_in_hours = convert_to_hours(op_time_interval)

        feature = pd.DataFrame([encoded_segment, shock_interval_in_hours, op_interval_in_hours,
                                encoded_typ, dataset.iloc[:, 3]],
                    index = ['Segment', 'Shock_interval', 'Operation_interval',
                             'Type', 'Shock_level']).transpose()
        concat_feature = pd.concat([intensity, feature], axis=1)

        # Drop NA rows
        features = concat_feature.dropna()
        return features