sweep_config = {
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
            # Log uniform search centered around 2e-5
            'distribution': 'log_uniform_values',
            'min': 5e-6,
            'max': 5e-5,
        },
        'epochs': {
            # Larger range of PPO update epochs
            'values': [3, 5, 8, 10]
        },
        'gamma': {
            # High gamma values for long time horizons
            'distribution': 'uniform',
            'min': 0.991,
            'max': 0.9999
        },
        'c1': {
            # Value function coefficient
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0
        },
        'c2': { 
            # Entropy coefficient
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.1
        }
    }
}
