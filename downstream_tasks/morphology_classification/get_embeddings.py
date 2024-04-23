import sys, os

sys.path.insert(os.path.abspath('/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino/dinov2/'))
from dinov2.eval.setup import setup_and_build_model

class config:
    """Configuration for the AstroDINO model."""
    output_dir = '/mnt/home/lparker/ceph/dino_training'
    config_file = '/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino_legacy/astrodino/configs/ssl_default_config.yaml'
    pretrained_weights = '/mnt/home/lparker/ceph/astrodino/vitl12_simplified_better_wd/training_199999/teacher_checkpoint.pth'
    opts = []

AstroDINO, dtype = setup_and_build_model(config())


