conf = {
    # "save_path": "C:/Users/rr/Desktop/sftp/mlp/trained-local",  # win
    # "load_path": "C:/Users/rr/Desktop/sftp/mlp/trained-local",
    "save_path": "/home/rr/Downloads/Export/trained/mlp",  # linux
    "load_path": "/home/rr/Downloads/Export/trained/mlp",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "CUDA_USE": [0, 1, 2, 3],
    # "data": r"C:\Users\rr\Desktop\documents\Export",  # win
    "data": "/home/rr/Downloads/Export",  # linux
    "model": {
        'model_name': 'MLP',
        'epoch': 150,
        'batch_size': 200,  # 1200
        'encoder_dim': 5307,
        'mlp_ratio': 4.,
        'encoder_dropout': 0.3,
        'decoder_dim': [5307, 4096, 2048, 1024, 618],
        'decoder_dropout': 0.3,
        'lr': 0.0001,
        'layer_num': 4
    },
}
