import datetime
import json

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

cassandra_endpoint = '10.62.1.118'
auth_provider = PlainTextAuthProvider(username='mlprw', password='q4RgwD$wK7*z')
cluster = Cluster([cassandra_endpoint], auth_provider=auth_provider)
mlp_session = cluster.connect('dev_mlpks')


def insert_model_info(model_id, version, file_name, desc, model_type, algorithm='XGBClassifier', hyper_parameter=None, eval_metrics=None, extended_att=None, features_dict=None):
    """Inserts model info into Cassandra table"""
    if features_dict is None:
        features_dict = {}
    if not extended_att:
        extended_att = "{}"

    mlp_session.execute(
        """
        INSERT INTO ml_model_storage (model_type, model_id, version, features_cat, features_encoded, features_num, repo_path, description, creation_date, modification_date, algorithm, hyper_parameter, eval_metrics, extended_attributes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (model_type,
         model_id,
         version,
         json.dumps(features_dict['FEATURES_CAT']),
         json.dumps(features_dict['FEATURES_ENCODED']),
         json.dumps(features_dict['FEATURES_NUM']),
         file_name,
         desc,
         datetime.datetime.utcnow(),
         datetime.datetime.utcnow(),
         algorithm,
         hyper_parameter,
         eval_metrics,
         extended_att)
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
