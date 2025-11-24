sweep_config = {
    
    'method': 'bayes',
    'metric': {
        'name': 'eval/mean_reward',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        'eta': 2
    },
    'parameters': {
        'learning_rate': {
        
            'distribution': 'uniform',
            'min': 2.0e-5,
            'max': 4.5e-5,
        },
        'epochs': {

            'values': [12, 14, 16]
        },
        'gamma': {

            'distribution': 'uniform',
            'min': 0.989,
            'max': 0.993
        },
        'c1': {

            'distribution': 'uniform',
            'min': 0.5,
            'max': 0.6
        },
        'c2': { 

            'distribution': 'log_uniform_values',
            'min': 0.001,
            'max': 0.01
        }
    }
}
