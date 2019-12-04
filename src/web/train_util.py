from src.web.preprocessing import PreProcessing
from src.web.preprocessing import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib
import os
from os import path

import datetime
import json
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
# import repositorytools
import time
from sklearn import cross_validation
from sklearn import metrics
from sklearn.dummy import DummyClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from shutil import copyfile

from src.web.repository import repository as repo
from src.web.repository import artifact as art


WORK_DIR = '/var/spark/ml_files/'
MODEL_DIR = WORK_DIR + "models"
cassandra_endpoint = '10.81.12.121' #'10.62.1.118'
auth_provider = PlainTextAuthProvider(username='mlprw', password='q4RgwD$wK7*z')
cluster = Cluster([cassandra_endpoint], auth_provider=auth_provider)
mlp_session = cluster.connect('dev_mlpks')


def write_model(model, model_name, idx=None):
    """Write the model into a file """
    build_id = "" if idx is None else "_" + str(idx)
    file_name = path.join(MODEL_DIR, '%s%s.pkl' % (model_name, build_id))
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    file = joblib.dump(model, file_name)
    return (file, file_name)

REPOSITORY_URL = 'http://nexus-master.digitalriverws.net/nexus/repository' #'http://10.48.48.10/nexus' #'http://nexus.digitalriverws.net/nexus'
REPO_USER = 'deployment'
REPO_PWD = 'deployment123'
REPO_ID = 'foundationreleases'
REPO_GROUP = 'com.digitalriver.prediction-service'

PREPROCESS_DIR = 'src/web/'

def insert_model_info(model_id, version, file_name, desc, model_type, algorithm='XGBClassifier', hyper_parameter=None, eval_metrics=None, extended_att=None, features_dict={}):
    """Inserts model info into Cassandra table"""
    if not extended_att:
        extended_att= "{}"

    mlp_session.execute(
        """
        INSERT INTO ml_model_storage (model_type, model_id, version, features_cat, features_encoded, features_num, repo_path, description, creation_date, modification_date, algorithm, hyper_parameter, eval_metrics, extended_attributes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (model_type, model_id, version, json.dumps(features_dict['FEATURES_CAT']), json.dumps(features_dict['FEATURES_ENCODED']), json.dumps(features_dict['FEATURES_NUM']), file_name, desc, datetime.datetime.utcnow(), datetime.datetime.utcnow(), algorithm, hyper_parameter, eval_metrics, extended_att)

    )
    print("Model %s version %d is inserted into model repo" % (model_id, version))


def get_latest_version(model_id, model_type):
    """Get latest version of the given model_id"""
    latest_version_query = "select version from ml_model_storage  where model_type = '%s' and model_id = '%s' limit 1" % (model_type, model_id)
    query_result = mlp_session.execute(latest_version_query).one()
    if query_result is None:
        latest_version = 0
    else:
        latest_version = query_result.version

    return latest_version


def get_repo_client():
    return repo.repository_client_factory(repository_url=REPOSITORY_URL, user=REPO_USER, password=REPO_PWD)


def upload_artifact(file_path):
    """Upload artifact to Nexus Repo"""
    artifact = art.LocalArtifact(local_path=file_path, group=REPO_GROUP)
    client = get_repo_client()
    remote_artifacts = client.upload_artifacts(local_artifacts=[artifact], repo_id=REPO_ID, use_direct_put=True, _path_prefix='')
    print(remote_artifacts)
    return str(remote_artifacts[0]) if remote_artifacts else ''


def handle_preprocessing_file(model_id, version):
    preprocess_file_name = '{}_{}_preprocessing.py'.format(model_id, version)
    preprocess_file_path = PREPROCESS_DIR + preprocess_file_name
    copyfile(PREPROCESS_DIR + "preprocessing.py", preprocess_file_path)
    repo_path = upload_artifact(preprocess_file_path)
    return repo_path

def create_preprocessing_file(model_id, version):
    preprocess_file_name = '{}_{}_preprocessing.py'.format(model_id, version)
    preprocess_file_path = PREPROCESS_DIR + preprocess_file_name
    copyfile(PREPROCESS_DIR + "preprocessing.py", preprocess_file_path)
    return preprocess_file_path

def get_feat_importances(pipe, model_name, df_features):
    classifier = pipe.named_steps[model_name]
    feature_importance = classifier.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    print("feature_importance column ",df_features.columns[sorted_idx])
    print("feature_importance val ",feature_importance[sorted_idx])
    print('sorted_idx: ', sorted_idx)
    feature_importance_vals = feature_importance[sorted_idx]
    feature_importance_columns = df_features.columns[sorted_idx]
    return feature_importance_columns, feature_importance_vals, sorted_idx

def display_feature_importance(pcols, pvals, sorted_idx):
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(8,12))
    plt.barh(pos, pvals, align='center')
    plt.yticks(pos, pcols)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')

def build_and_train(df, clf, param_grid, model_name, model_file = '', best_param=None, features_dict={}, test_data=None):
    model_prefix = model_name + '__'
    time_start = time.time()
    df_X = df[features_dict['FIELDS']]
    LABEL = features_dict['LABEL']


    bin_profile = features_dict.get('df_bin_profile', None)
    #     bank_profile = features_dict['df_bank_profile']
    result_dict = {}
    if test_data is None:
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(df_X, df[LABEL], \
                                                                             test_size=0.1, random_state=42)
    else:
        print("# Using assigned test_data")
        x_train, x_test, y_train, y_test = df_X, test_data[features_dict['FIELDS']], df[LABEL], test_data[LABEL]

    y_pred_train = None
    print("--------------------")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    if bin_profile is not None:
        print('## bin_profile: {}'.format(bin_profile.shape))

    #     imputer =  Imputer(strategy="most_frequent",missing_values=0, axis=0)
    if best_param is None:

        #         pipe = make_pipeline(PreProcessing(bin_profile), imputer, clf())
        pipe = make_pipeline(PreProcessing(bin_profile), clf())
        score = 'average_precision'#'f1' #'accuracy' #  ['accuracy','precision_macro', 'recall_macro', 'f1_macro']
        print("# Tuning hyper-parameters for %s" % score)

        pipe_param_grid = {model_prefix + k: v for k, v in param_grid.items()}
        print("pipe_param_grid ", pipe_param_grid)

        clf_gs = GridSearchCV(sc, pipe, pipe_param_grid, cv=3, scoring=score, n_jobs=-1, fit_params={features_dict_key: features_dict})
        #         clf_gs = GridSearchCV(pipe, pipe_param_grid, cv=3, scoring=score, n_jobs=-1, fit_params={features_dict_key: features_dict})
        clf_gs = clf_gs.fit(x_train, y_train)

        print('clf_gs: {}'.format(clf_gs))
        best_model = clf_gs.best_estimator_  #clf_gs.estimator.named_steps[model_name]
        print('best params: {}'.format(best_model.get_params(deep=True)))
        #         result_dict['hyper_params'] = best_model.get_params(deep=True) #clf_gs.best_params_
        print("# Grid scores on development set:")
        means = clf_gs.cv_results_['mean_test_score']
        stds = clf_gs.cv_results_['std_test_score']
        zipped = list(zip(means, stds, clf_gs.cv_results_['params']))
        for mean, std, params in sorted(zipped, key = lambda x: x[0]) :
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

        best_parameters = best_model.get_params(deep=True)
        print("# Best parameters set found on development set: {}".format(best_parameters))

        best_parameters = {k.replace(model_prefix,''): v for k, v in best_parameters.items()}
        print('best params: {}'.format(best_parameters))
        pipe = best_model
        y_pred_train = pipe.predict(x_train).round()
    else:
        best_parameters= best_param
        #         pipe = make_pipeline(PreProcessing(bin_profile), imputer, clf(**best_parameters))
        pipe = make_pipeline(PreProcessing(bin_profile), clf(**best_parameters))


    if y_pred_train is None:
        if model_file != '':
            best_parameters['updater'] = 'refresh'
            best_parameters['refresh_leaf'] = True


        # if model_name == 'xgbclassifier':
        #     print("training xgb ....... ")
        #     eval_metric = features_dict.get('eval_metric', 'aucpr')
        #     early_stopping_rounds = features_dict.get('early_stopping_rounds', None)
        #     eval_set = features_dict.get('eval_set', None)
        #
        #     if model_file == '':
        #         if early_stopping_rounds:
        #             y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict, xgbclassifier__eval_metric=eval_metric, xgbclassifier__eval_set=eval_set, xgbclassifier__early_stopping_rounds=early_stopping_rounds).round()
        #         else:
        #             y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict).round()
        #
        #     else:
        #         print("Using model_file to train: ", model_file)
        #         y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict, xgbclassifier__eval_metric=eval_metric).round()
        # elif model_name == 'catboostclassifier':
        #     print("training catboostclassifier ....... ")
        #     fit_params = features_dict.get('fit_params', None)
        #     if fit_params:
        #         print("using fit_params ....... ")
        #         y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict, **fit_params).round()
        #     else:
        #         y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict).round()
        fit_params = features_dict.get('fit_params', None)
        if fit_params:
            print("using fit_params ....... ")
            y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict, **fit_params).round()
        else:
            y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict).round()

        # else:
        #     print("training other  ....... ")
        #     y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict).round()

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
    #     print_accuracy_report(pipe, x_train, y_train, num_validations=3)
    #     print(metrics.classification_report(y_test, y_pred_test))
    test_class_report = metrics.classification_report(y_test, y_pred_test)
    result_dict['train_class_report'] = train_class_report
    result_dict['test_class_report'] = test_class_report
    conf_mx = metrics.confusion_matrix(y_test, y_pred_test)

    preprocess = pipe.named_steps['preprocessing']
    x_encoded = preprocess.encode(x_train[:1])
    df_features = x_encoded[preprocess.features_all]

    feat_importance_cols, feat_importance_vals, sorted_idx =  get_feat_importances(pipe, model_name, df_features)
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

    return pipe, result_dict
