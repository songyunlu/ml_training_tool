
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
import repositorytools

WORK_DIR = '/var/spark/ml_files/'
MODEL_DIR = WORK_DIR + "models"
MODEL_TYPE = "ML-BR"
cassandra_endpoint = '10.62.1.118'
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

REPOSITORY_URL = 'http://nexus.digitalriverws.net/nexus'
REPO_USER = 'deployment'
REPO_PWD = 'deployment123'
REPO_ID = 'foundationreleases'
REPO_GROUP = 'com.digitalriver.prediction-service'

PREPROCESS_DIR = 'src/web/'

def insert_model_info(model_id, version, file_name, desc, model_type=MODEL_TYPE, algorithm='XGBClassifier', hyper_parameter=None, eval_metrics=None, extended_att=None):
    """Inserts model info into Cassandra table"""
    if not extended_att:
        extended_att= "{}"
    
    mlp_session.execute(
    """
    INSERT INTO ml_model_storage (model_type, model_id, version, features_cat, features_encoded, features_num, repo_path, description, creation_date, modification_date, algorithm, hyper_parameter, eval_metrics, extended_attributes)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """,
    (model_type, model_id, version, json.dumps(FEATURES_CAT), json.dumps(FEATURES_ENCODED), json.dumps(FEATURES_NUM), file_name, desc, datetime.datetime.utcnow(), datetime.datetime.utcnow(), algorithm, hyper_parameter, eval_metrics, extended_att)
        
    )
    print("Model %s version %d is inserted into model repo" % (model_id, version))      

    
def get_latest_version(model_id, model_type=MODEL_TYPE):
    """Get latest version of the given model_id"""
    latest_version_query = "select version from ml_model_storage  where model_type = '%s' and model_id = '%s' limit 1" % (model_type, model_id)
    query_result = mlp_session.execute(latest_version_query).one()
    if query_result is None:
        latest_version = 0
    else:
        latest_version = query_result.version
    
    return latest_version


def upload_artifact(file_path):
    """Upload artifact to Nexus Repo"""
    artifact = repositorytools.LocalArtifact(local_path=file_path, group=REPO_GROUP)

    client = repositorytools.repository_client_factory(repository_url=REPOSITORY_URL, user=REPO_USER, password=REPO_PWD)
    remote_artifacts = client.upload_artifacts(local_artifacts=[artifact], repo_id=REPO_ID, use_direct_put=True)
    print(remote_artifacts)
    return str(remote_artifacts[0]) if remote_artifacts else ''


def handle_preprocessing_file(model_id, version) :
    from shutil import copyfile
    preprocess_file_name = '{}_{}_preprocessing.py'.format(model_id, version)
    preprocess_file_path = PREPROCESS_DIR + preprocess_file_name
    copyfile(PREPROCESS_DIR + "preprocessing.py", preprocess_file_path)
    repo_path = upload_artifact(preprocess_file_path)
    return repo_path
    


