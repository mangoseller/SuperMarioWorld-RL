sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'eval/mean_reward',
        'goal': 'maximize'
    },
    'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            's': 2,
            'eta': 3,
            'max_iter': 16

    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'c2': { # Entropy
               'distribution': 'uniform',
               'min': 0.01,
               'max': 0.1
               },
        # 'gamma': {
        #     'values': [0.99, 0.995, 0.999]
        # },
        # 'minibatch_size': {
        #     'values': [32, 64, 128]
        # },
        # 'epochs': {
        #     'values': [3, 4, 5]
        # }
    }
}
