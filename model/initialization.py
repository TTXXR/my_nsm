import os
from .utils import load_data
from .model import Model


def initialize_model(config, train_source, test_source):
    print("Initializing model...")
    model_config = config['model']
    model_config['save_path'] = config['save_path']
    model_config['load_path'] = config['load_path']
    model_config['train_source'] = train_source
    model_config['test_source'] = test_source
    model = Model(**model_config)
    print("Model initialization complete.")
    return model


def initialization(config, train=False, test=False):
    print("Initializing...")
    save_path = config['save_path']
    os.chdir(save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    print("Initializing data source...")
    train_source = load_data(os.path.join(config['data'], "Train"), cache=(train or test))
    # test_source = load_data(os.path.join(config['data'], "Test"), cache=(train or test))
    test_source = train_source
    print("Data initialization complete.")
    return initialize_model(config, train_source, test_source)
