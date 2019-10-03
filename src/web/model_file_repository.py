import repositorytools

REPOSITORY_URL = 'http://nexus.digitalriverws.net/nexus'
REPO_USER = 'deployment'
REPO_PWD = 'deployment123'
REPO_ID = 'foundationreleases'
REPO_GROUP = 'com.digitalriver.prediction-service'
PREPROCESS_DIR = 'src/web/'


def handle_preprocessing_file(model_id, version):
    from shutil import copyfile
    preprocess_file_name = '{}_{}_preprocessing.py'.format(model_id, version)
    preprocess_file_path = PREPROCESS_DIR + preprocess_file_name
    copyfile(PREPROCESS_DIR + "preprocessing.py", preprocess_file_path)
    repo_path = upload_artifact(preprocess_file_path)
    return repo_path


def upload_artifact(file_path):
    """Upload artifact to Nexus Repo"""
    artifact = repositorytools.LocalArtifact(local_path=file_path, group=REPO_GROUP)

    client = repositorytools.repository_client_factory(repository_url=REPOSITORY_URL, user=REPO_USER, password=REPO_PWD)
    remote_artifacts = client.upload_artifacts(local_artifacts=[artifact], repo_id=REPO_ID, use_direct_put=True)
    print(remote_artifacts)
    return str(remote_artifacts[0]) if remote_artifacts else ''
