from ducho.runner.Runner import MultimodalFeatureExtractor
import torch
import os
import numpy as np
import random

def set_seed(seed = 42):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


def main():
    set_seed()
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo_baby/config.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main()
