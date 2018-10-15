#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Naive Bayes implementation.
Maximizes the log likelihood to prevent underflow,
and applies Laplace smoothing to solve the zero observations problem.

API inspired by SciKit-learn.

Sources:
  * https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
  * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py
  * https://github.com/ashkonf/HybridNaiveBayes
"""

from datasets import load_loan_defaulters
from feature import ContinuousFeature
from feature import DiscreteFeature
from naive_bayes import NaiveBayes


def main():
    dataset = load_loan_defaulters()
    # dataset = load_load_dataset("creditcard.csv")
    design_matrix = [row[:-1] for row in dataset]
    target_values = [row[-1] for row in dataset]
    clf = NaiveBayes(extract_features)
    clf.fit(design_matrix, target_values)
    scores = cross_val_score(clf, df, y, cv=10)
    prediction = clf.predict_record([0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 15998980980.64, 0])
    print("Credit Card Category: ",prediction)


def extract_features(feature_vector):
    """Maps a feature vector to whether each feature is continuous or discrete."""
    return [
        DiscreteFeature(feature_vector[0]),
        ContinuousFeature(feature_vector[1]),
        ContinuousFeature(feature_vector[2]),
        ContinuousFeature(feature_vector[3]),
        ContinuousFeature(feature_vector[4]),
        ContinuousFeature(feature_vector[5]),
        ContinuousFeature(feature_vector[6]),
        ContinuousFeature(feature_vector[7]),
        ContinuousFeature(feature_vector[8]),
        ContinuousFeature(feature_vector[9]),
        ContinuousFeature(feature_vector[10]),
        ContinuousFeature(feature_vector[11]),
        ContinuousFeature(feature_vector[12]),
        ContinuousFeature(feature_vector[13]),
        ContinuousFeature(feature_vector[14]),
        ContinuousFeature(feature_vector[15]),
        ContinuousFeature(feature_vector[16]),
        ContinuousFeature(feature_vector[17]),
        ContinuousFeature(feature_vector[18]),
        ContinuousFeature(feature_vector[19]),
        ContinuousFeature(feature_vector[20]),
        ContinuousFeature(feature_vector[21]),
        ContinuousFeature(feature_vector[22]),
        ContinuousFeature(feature_vector[23]),
        ContinuousFeature(feature_vector[24]),
        ContinuousFeature(feature_vector[25]),
        ContinuousFeature(feature_vector[26]),
        ContinuousFeature(feature_vector[27]),
        ContinuousFeature(feature_vector[28]),
        ContinuousFeature(feature_vector[29])
    ]


if __name__ == '__main__':
    main()
