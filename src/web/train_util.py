import os
import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

from src.web.preprocessing import PreProcessing
from src.web.preprocessing import make_pipeline

WORK_DIR = '/var/spark/ml_files/'
MODEL_DIR = WORK_DIR + "models"


def write_model(model, model_name, idx=None):
    """Write the model into a file """
    build_id = "" if idx is None else "_" + str(idx)
    file_name = path.join(MODEL_DIR, '%s%s.pkl' % (model_name, build_id))
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    file = joblib.dump(model, file_name)
    return file, file_name


def get_feat_importances(pipe, model_name, df_features):
    classifier = pipe.named_steps[model_name]
    feature_importance = classifier.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    print("feature_importance column ", df_features.columns[sorted_idx])
    print("feature_importance val ", feature_importance[sorted_idx])
    print('sorted_idx: ', sorted_idx)
    feature_importance_vals = feature_importance[sorted_idx]
    feature_importance_columns = df_features.columns[sorted_idx]
    return feature_importance_columns, feature_importance_vals, sorted_idx


def display_feature_importance(pcols, pvals, sorted_idx):
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(8, 12))
    plt.barh(pos, pvals, align='center')
    plt.yticks(pos, pcols)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')


def build_and_train(df, clf, param_grid, model_name, model_file='', best_param=None, features_dict=None, metrics_feedback_url=None):
    if features_dict is None:
        features_dict = {}

    model_prefix = model_name + '__'
    time_start = time.time()
    df_X = df[features_dict['FIELDS']]
    LABEL = features_dict['LABEL']

    print(features_dict)
    bin_profile = features_dict['df_bin_profile']
    print("===========================")
    # bank_profile = features_dict['df_bank_profile']
    result_dict = {}
    x_train, x_test, y_train, y_test = train_test_split(df_X, df[LABEL], test_size=0.1, random_state=42)
    y_pred_train = None
    print("--------------------")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    if bin_profile is not None:
        print('## bin_profile: {}'.format(bin_profile.shape))

    if best_param is None:
        pipe = make_pipeline(PreProcessing(bin_profile), Imputer(strategy="most_frequent", axis=0), clf())
        # pipe = make_pipeline(PreProcessing(bin_profile), clf())
        score = 'average_precision'  # 'f1' #'accuracy' #  ['accuracy','precision_macro', 'recall_macro', 'f1_macro']
        print("# Tuning hyper-parameters for %s" % score)

        pipe_param_grid = {model_prefix + k: v for k, v in param_grid.items()}
        print("pipe_param_grid ", pipe_param_grid)

        clf_gs = GridSearchCV(sc, pipe, pipe_param_grid, cv=3, scoring=score, n_jobs=-1, fit_params={features_dict_key: features_dict})
        # clf_gs = GridSearchCV(pipe, pipe_param_grid, cv=3, scoring=score, n_jobs=-1, fit_params={features_dict_key: features_dict})
        clf_gs = clf_gs.fit(x_train, y_train)

        print('clf_gs: {}'.format(clf_gs))
        best_model = clf_gs.best_estimator_  # clf_gs.estimator.named_steps[model_name]
        print('best params: {}'.format(best_model.get_params(deep=True)))
        # result_dict['hyper_params'] = best_model.get_params(deep=True) #clf_gs.best_params_
        print("# Grid scores on development set:")
        means = clf_gs.cv_results_['mean_test_score']
        stds = clf_gs.cv_results_['std_test_score']
        zipped = list(zip(means, stds, clf_gs.cv_results_['params']))
        for mean, std, params in sorted(zipped, key=lambda x: x[0]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        best_parameters = best_model.get_params(deep=True)
        print("# Best parameters set found on development set: {}".format(best_parameters))

        best_parameters = {k.replace(model_prefix, ''): v for k, v in best_parameters.items()}
        print('best params: {}'.format(best_parameters))
        pipe = best_model
        y_pred_train = pipe.predict(x_train).round()
    else:
        best_parameters = best_param
        pipe = make_pipeline(PreProcessing(bin_profile), Imputer(strategy="most_frequent", axis=0), clf(**best_parameters))

    if y_pred_train is None:
        if model_file != '':
            best_parameters['updater'] = 'refresh'
            best_parameters['refresh_leaf'] = True

        if model_name == 'xgbclassifier':
            print("training xgb ....... ")
            if model_file == '':
                eval_metric = features_dict.get('eval_metric', 'error')
                eval_set = [(x_test, y_test)]

                y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict, xgbclassifier__eval_metric=eval_metric).round()

            else:
                print("Using model_file to train: ", model_file)
                y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict, xgbclassifier__xgb_model=model_file, xgbclassifier__eval_metric='error').round()

        else:
            y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict).round()

    result_dict['hyper_params'] = best_parameters
    print("best_parameters ", best_parameters)
    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("x_train  ..... ", x_train.head(5))

    print("pipe:", pipe)
    train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
    print("training accuracy:", train_accuracy)

    train_auc = metrics.roc_auc_score(y_train, y_pred_train)
    print("training auc:", train_auc)

    train_class_report = metrics.classification_report(y_train, y_pred_train)
    print(train_class_report)

    y_pred_test = pipe.predict(x_test).round()

    training_time = time.time() - time_start
    print("# training time:", training_time)
    result_dict['training_time'] = training_time

    '''Dummy classifier for comparison'''
    clf_d = DummyClassifier(strategy='most_frequent')
    clf_d.fit(np.zeros_like(x_train), y_train.astype(np.float))
    y_pred_test_dummy = clf_d.predict(np.zeros_like(x_test)).round()
    accuracy_dummy = metrics.accuracy_score(y_test, y_pred_test_dummy)
    '''End of Dummy Classifier'''

    test_accuracy = metrics.accuracy_score(y_test, y_pred_test)

    result_dict['accuracy_dummy'] = accuracy_dummy
    result_dict['train_accuracy'] = train_accuracy
    result_dict['test_accuracy'] = test_accuracy
    # print_accuracy_report(pipe, x_train, y_train, num_validations=3)
    # print(metrics.classification_report(y_test, y_pred_test))
    test_class_report = metrics.classification_report(y_test, y_pred_test)
    result_dict['train_class_report'] = train_class_report
    result_dict['test_class_report'] = test_class_report
    conf_mx = metrics.confusion_matrix(y_test, y_pred_test)

    preprocess = pipe.named_steps['preprocessing']
    print('x_train: {}'.format(x_train.head()))
    df_features = x_train[preprocess.features_all]

    feat_importance_cols, feat_importance_vals, sorted_idx = get_feat_importances(pipe, model_name, df_features)
    display_feature_importance(feat_importance_cols, feat_importance_vals, sorted_idx)

    result_dict['feature_importance_columns'] = str(feat_importance_cols)
    result_dict['feature_importance_vals'] = str(feat_importance_vals)
    result_dict['sorted_idx'] = str(sorted_idx)

    test_auc = metrics.roc_auc_score(y_test, y_pred_test)
    result_dict['train_auc'] = train_auc
    result_dict['test_auc'] = test_auc

    print("accuracy_dummy .....:", accuracy_dummy)

    print("test accuracy:", test_accuracy)
    print("test auc:", test_auc)
    print(test_class_report)

    print("# confusion_matrix -  test:\n", conf_mx)
    result_dict['conf_mx'] = conf_mx.tolist()

    save_metrics(metrics_feedback_url, {
        'type': 'CROSS_VALIDATION',
        'accuracy': train_accuracy,
        'rocAuc': train_auc
    })
    save_metrics(metrics_feedback_url, {
        'type': 'TESTING',
        'accuracy': test_accuracy,
        'rocAuc': test_auc,
        'confusionMatrix': np.array2string(conf_mx, separator=',')
    })

    return pipe, result_dict


def save_metrics(url=None, payload=None):
    if payload is None:
        payload = {}

    if url is not None:
        requests.post(url, json=payload)
