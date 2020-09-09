import datetime
import io
import json
import os
import sys
import time
from importlib import import_module
from os import path
from pickle import HIGHEST_PROTOCOL
from shutil import copyfile

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from src.web.preprocessing import PreProcessing
from src.web.preprocessing import make_pipeline
from src.web.repository import artifact as art
from src.web.repository import repository as repo

from os import path

WORK_DIR = '/var/spark/ml_files/'
MODELS_PATH = "models"
MODEL_DIR = WORK_DIR + MODELS_PATH
bucket = 'dr-machine-learning-sandbox'
s3 = boto3.resource('s3')

S3_BUCKET = 'dr-billing-opt'
S3_TRAIN_DIR = 'training_files'

cassandra_endpoint = '10.81.12.121'  # '10.62.1.118'
auth_provider = PlainTextAuthProvider(username='mlprw', password='q4RgwD$wK7*z')
try:
    cluster = Cluster([cassandra_endpoint], auth_provider=auth_provider)
    mlp_session = cluster.connect('dev_mlpks')
except:
    print('cannot connect to cassandra')


def read_from(file_path, usecols=None, s3_dir=None):
    if path.exists(file_path):
        obj = file_path
    else:
        s3 = boto3.client('s3')
        if s3_dir:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{s3_dir}/{file_path}')
        else:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{S3_TRAIN_DIR}/{file_path}')
        obj = io.BytesIO(obj['Body'].read())

    if usecols:
        df = pd.read_csv(obj, usecols=usecols)
    else:
        df = pd.read_csv(obj)

    return df


def write_model(model, model_name, idx=None):
    """Write the model into a file """
    build_id = "" if idx is None else "_" + str(idx)
    file_name = path.join(MODEL_DIR, '%s%s.pkl' % (model_name, build_id))
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    file = joblib.dump(model, file_name, protocol=HIGHEST_PROTOCOL)
    return file, file_name


def write_model_s3(model, model_name, idx=None):
    """Write the model into a file and stores to s3 """
    print('Writing file to S3')
    build_id = "" if idx is None else "_" + str(idx)
    _file_name = f'{MODELS_PATH}/{model_name}{build_id}.pkl'
    file = joblib.dump(model, _file_name, protocol=HIGHEST_PROTOCOL)
    response = s3.Bucket(bucket).upload_file(_file_name, f'billing-optimization/models/{model_name}{build_id}.pkl')
    file_name = f's3://{bucket}/{_file_name}'
    print(f'Uploaded to S3. Response {response}')
    return (file, file_name)


REPOSITORY_URL = 'http://nexus-master.digitalriverws.net/nexus/repository'  # 'http://10.48.48.10/nexus' #'http://nexus.digitalriverws.net/nexus'
REPO_USER = 'deployment'
REPO_PWD = 'deployment123'
REPO_ID = 'foundationreleases'
REPO_GROUP = 'com.digitalriver.prediction-service'

PREPROCESS_DIR = 'src/web/'


def insert_model_info(model_id, version, file_name, desc, model_type, algorithm='XGBClassifier', hyper_parameter=None, eval_metrics=None, extended_att=None, features_dict={}):
    """Inserts model info into Cassandra table"""
    if not extended_att:
        extended_att = "{}"

    mlp_session.execute(
        """
        INSERT INTO ml_model_storage (model_type, model_id, version, features_cat, features_encoded, features_num, repo_path, description, creation_date, modification_date, algorithm, hyper_parameter, eval_metrics, extended_attributes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (model_type, model_id, version, json.dumps(features_dict['FEATURES_CAT']), json.dumps(features_dict['FEATURES_ENCODED']), json.dumps(features_dict['FEATURES_NUM']), file_name, desc,
         datetime.datetime.utcnow(), datetime.datetime.utcnow(), algorithm, hyper_parameter, eval_metrics, extended_att)

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
    return preprocess_file_name, preprocess_file_path


def handle_preprocessing_file_s3(model_id, version):
    preprocess_file_name, preprocess_file_path = create_preprocessing_file(model_id, version)
    response = s3.Bucket(bucket).upload_file(preprocess_file_path, f'billing-optimization/models/{preprocess_file_name}')
    file_name = f's3://{bucket}/{preprocess_file_name}'
    print(f'Uploaded to S3. Response {response}')
    return file_name


def get_feat_importances(pipe, alg_name, df_features):
    classifier = pipe.named_steps[alg_name]
    feature_importance = classifier.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    print("feature_importance column ", df_features.columns[sorted_idx])
    print("feature_importance val ", feature_importance[sorted_idx])
    print('sorted_idx: ', sorted_idx)
    feature_importance_vals = feature_importance[sorted_idx]
    feature_importance_columns = df_features.columns[sorted_idx]
    return feature_importance_columns, feature_importance_vals, sorted_idx


def display_feature_importance(pcols, pvals, sorted_idx, model_name=None, output_dir=None):
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(8, 12))
    plt.barh(pos, pvals, align='center')
    plt.yticks(pos, pcols)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.tight_layout()
    if output_dir:
        plt.savefig(f'{output_dir}/feature_importance.png')
    elif model_name:
        plt.savefig(f'src/web/feat_importance/{model_name}_feature_importance.png')


def roc_auc(y, p):
    fpr, tpr, threshold = metrics.roc_curve(y, p)
    return metrics.auc(fpr, tpr)


def save_feature_importance(pcols, pvals, output_dir=None):
    feature_importance = {}
    for i in range(len(pcols)):
        feature_importance[pcols[i]] = pvals[i]

    if output_dir:
        with open(f'{output_dir}/feature.importance', 'w') as feature_importance_out:
            json.dump(feature_importance, feature_importance_out, ensure_ascii=False, indent=4)


def save_metrics(metrics_type=None, metrics_payload=None, output_dir=None):
    if metrics_payload is None:
        metrics_payload = {}

    if output_dir and metrics_type is not None:
        with open(f'{output_dir}/{metrics_type}.metrics', 'w') as metrics_json:
            json.dump(metrics_payload, metrics_json, ensure_ascii=False, indent=4)


def build_and_train(df, clf, param_grid, alg_name, model_file='', best_param=None, features_dict={}, test_data=None, output_dir=None):
    if features_dict is None:
        features_dict = {}

    print(features_dict)

    model_prefix = alg_name + '__'
    time_start = time.time()
    df_X = df[features_dict['FIELDS']]
    LABEL = features_dict['LABEL']

    result_dict = {}
    if test_data is None:
        x_train, x_test, y_train, y_test = train_test_split(df_X, df[LABEL], \
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

    preprocessing_file = 'src.web.preprocessing'
    import_module(preprocessing_file)
    sys.modules['src.web.preprocessing'] = sys.modules[preprocessing_file]

    #     imputer =  Imputer(strategy="most_frequent",missing_values=0, axis=0)
    if best_param is None:

        #         pipe = make_pipeline(PreProcessing(), imputer, clf())
        pipe = make_pipeline(PreProcessing(), clf())
        score = 'average_precision'  # 'f1' #'accuracy' #  ['accuracy','precision_macro', 'recall_macro', 'f1_macro']
        print("# Tuning hyper-parameters for %s" % score)

        pipe_param_grid = {model_prefix + k: v for k, v in param_grid.items()}
        print("pipe_param_grid ", pipe_param_grid)

        # clf_gs = GridSearchCV(sc, pipe, pipe_param_grid, cv=3, scoring=score, n_jobs=-1, fit_params={'features_dict_key': features_dict})
        clf_gs = GridSearchCV(pipe, pipe_param_grid, cv=3, scoring=score, n_jobs=-1, fit_params={'features_dict_key': features_dict})

        # clf_gs = GridSearchCV(pipe, pipe_param_grid, cv=3, scoring=score, n_jobs=-1)

        fit_params = features_dict.get('fit_params', None)

        # clf_gs = clf_gs.fit(x_train, y_train, preprocessing__features_dict=features_dict, **fit_params)
        clf_gs = clf_gs.fit(x_train, y_train, preprocessing__features_dict=features_dict, **fit_params)

        print('clf_gs: {}'.format(clf_gs))
        best_model = clf_gs.best_estimator_  # clf_gs.estimator.named_steps[alg_name]
        print('best params: {}'.format(best_model.get_params(deep=True)))
        #         result_dict['hyper_params'] = best_model.get_params(deep=True) #clf_gs.best_params_
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
        #         pipe = make_pipeline(PreProcessing(), imputer, clf(**best_parameters))
        pipe = make_pipeline(PreProcessing(), clf(**best_parameters))

    if y_pred_train is None:
        # if model_file != '':
        #     best_parameters['updater'] = 'refresh'
        #     best_parameters['refresh_leaf'] = True

        fit_params = features_dict.get('fit_params', None)
        if fit_params:
            print("using fit_params ....... ")
            y_pred_train = pipe.fit_predict(x_train, y_train, preprocessing__features_dict=features_dict, **fit_params).round()
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

    y_proba_train = pipe.predict_proba(x_train)
    y_proba_train_array = np.asarray(list(map(lambda x: x[1], y_proba_train.tolist())))
    train_roc_auc = roc_auc(y_train, y_proba_train_array)

    # result_dict['train_auc'] = train_auc
    result_dict['train_auc'] = train_roc_auc

    print("training auc:", train_auc)
    print("training ROC AUC:", train_roc_auc)

    train_class_report = metrics.classification_report(y_train, y_pred_train)
    print(train_class_report)


    y_pred_test = pipe.predict(x_test)
    y_proba_test = pipe.predict_proba(x_test)
    y_proba_test_array = np.asarray(list(map(lambda x: x[1], y_proba_test.tolist())))

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

    test_class_report = metrics.classification_report(y_test, y_pred_test)
    result_dict['train_class_report'] = train_class_report
    result_dict['test_class_report'] = test_class_report
    conf_mx = metrics.confusion_matrix(y_test, y_pred_test)

    preprocess = pipe.named_steps['preprocessing']
    x_encoded = preprocess.encode(x_train[:1])
    df_features = x_encoded[preprocess.features_all]

    feat_importance_cols, feat_importance_vals, sorted_idx = get_feat_importances(pipe, alg_name, df_features)
    model_name = features_dict.get('model_name', None)
    display_feature_importance(feat_importance_cols, feat_importance_vals, sorted_idx, model_name, output_dir)
    save_feature_importance(feat_importance_cols, feat_importance_vals, output_dir)

    result_dict['feature_importance_columns'] = str(feat_importance_cols)
    result_dict['feature_importance_vals'] = str(feat_importance_vals)
    result_dict['sorted_idx'] = str(sorted_idx)

    test_auc = metrics.roc_auc_score(y_test, y_pred_test)
    test_roc_auc = roc_auc(y_test, y_proba_test_array)

    result_dict['train_auc'] = train_auc
    result_dict['test_auc'] = test_auc
    result_dict['train_ROC_AUC'] = train_roc_auc
    result_dict['test_ROC_AUC'] = test_roc_auc

    print("accuracy_dummy .....:", accuracy_dummy)

    print("test accuracy:", test_accuracy)
    print("test auc:", test_auc)
    print("test ROC AUC:", test_roc_auc)

    print(test_class_report)
    print("# confusion_matrix -  test:\n", conf_mx)
    result_dict['conf_mx'] = conf_mx.tolist()

    y_pred_scores = pipe.predict_proba(x_test)
    y_scores = np.asarray(list(map(lambda x: x[1], y_pred_scores.tolist())))
    pr_auc = metrics.average_precision_score(y_test, y_scores)
    print(f"PR AUC: {pr_auc}")

    result_dict['PR_AUC'] = pr_auc

    save_metrics(
        'cross_validation',
        {
            'type': 'CROSS_VALIDATION',
            'accuracy': train_accuracy,
            'rocAuc': train_roc_auc
        },
        output_dir
    )
    save_metrics(
        'testing',
        {
            'type': 'TESTING',
            'accuracy': test_accuracy,
            'rocAuc': test_roc_auc,
            'confusionMatrix': np.array2string(conf_mx, separator=','),
            'prAuc': pr_auc
        },
        output_dir
    )

    return pipe, result_dict
