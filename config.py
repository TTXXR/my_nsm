conf = {
    "save_path": r"F:\trained",
    "load_path": r"F:\trained",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_USE": [0],
    "data": r"F:\AI4Animation-master\AI4Animation\SIGGRAPH_Asia_2019\Export",
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
