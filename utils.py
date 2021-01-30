import os
import pandas as pd


def get_dataset_path(filename):
    dirname, _ = os.path.split(os.path.abspath(__file__))
    dataset_name = dirname + '\\' + filename
    return dataset_name


def preprocess_intensity_metric(intensity_data):
    intensity_feature = []
    for (idx, items) in intensity_data.iteritems():
        items = items.replace("'", "").replace('g', '').replace('/', '').translate(str.maketrans({"'": None})).split()
        conv_list = [float(i) for i in items]
        intensity_feature.append(tuple(conv_list))
    return pd.DataFrame(intensity_feature, columns=['Intensity_1', 'Intensity_2'])


def convert_to_numeric_features(feature):
    return feature.cat.codes


def calc_time_difference(time1, time2):
    return pd.to_timedelta(time1 - time2)


def convert_to_hours(time):
    return time.dt.total_seconds() / (60 * 60)
