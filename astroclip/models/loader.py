import joblib
from huggingface_hub import hf_hub_download


def load_model(repo_id, filename):
    model = joblib.load(hf_hub_download(repo_id=repo_id, filename=filename))
    return model
