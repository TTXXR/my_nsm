conf = {
    "save_path": "/home/ubuntu/rentianxiang/NSM/trained-tese",
    "load_path": "/home/ubuntu/rentianxiang/NSM/trained-test",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "CUDA_USE": [0, 1, 2, 3],
    "data": "/home/ubuntu/rentianxiang/NSM",
    "model": {
        'model_name': 'MLP',
        'epoch': 150,
        'batch_size': 1200,
        'encoder_dim': 5307,
        'mlp_ratio': 4.,
        'encoder_dropout': 0.3,
        'decoder_dim': [5307, 4096, 2048, 1024, 618],
        'decoder_dropout': 0.3,
        'lr': 0.0001,
    },
}
