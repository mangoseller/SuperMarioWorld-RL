IMPALA_SWEEP = {
    'method': 'bayes',
    'metric': {
        'name': 'eval/mean_reward',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'eta': 2
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 5e-4,
        },
        'epochs': {
            'values': [3, 4, 5, 6]
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.985,
            'max': 0.995
        },
        'c1': {
            'distribution': 'uniform',
            'min': 0.4,
            'max': 0.7
        },
        'c2': {
            'distribution': 'log_uniform_values',
            'min': 0.005,
            'max': 0.03
        }
    }
}

CONV_SWEEP = {
    'method': 'bayes',
    'metric': {
        'name': 'eval/mean_reward',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'eta': 2
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 5e-5,
        },
        'epochs': {
            'values': [6, 8, 10, 12]
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.989,
            'max': 0.996
        },
        'c1': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 0.9
        },
        'c2': {
            'distribution': 'log_uniform_values',
            'min': 0.005,
            'max': 0.02
        }
    }
}

IMPALA_LARGE_SWEEP = {
    'method': 'bayes',
    'metric': {
        'name': 'eval/mean_reward',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 2,
        'eta': 3,
    },
    'parameters': {
        # Learning rate is highest impact for large models
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 8e-5,
            'max': 2.5e-4,
        },
        # Entropy coefficient - exploration vs exploitation
        'c2': {
            'distribution': 'log_uniform_values',
            'min': 0.01,
            'max': 0.03,
        },
    }
}
