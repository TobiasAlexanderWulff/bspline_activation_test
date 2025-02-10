def get_benchmark_config():
    return {
        "model": ["cnn", "resnet", "densenet"],
        "activation": ["relu", "swish", "bspline"],
        "dataset": ["mnist", "cifar10", "cifar100"],
        "optimizer": ["adam"],
        "batch_size": [64, 128],
        "epochs": [10, 20],
        "learning_rate": [0.001, 0.01],
        "seed": [14, 42, 56],
    }