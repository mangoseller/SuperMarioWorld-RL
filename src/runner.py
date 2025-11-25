import argparse
from models import ConvolutionalSmall, ImpalaLike, ImpalaLarge
from config import (
    IMPALA_TRAIN_CONFIG, 
    IMPALA_TEST_CONFIG, 
    IMPALA_TUNE_CONFIG,
    IMPALA_LARGE_TRAIN_CONFIG,
    IMPALA_LARGE_TEST_CONFIG,
    IMPALA_LARGE_TUNE_CONFIG,
    CONV_TRAIN_CONFIG, 
    CONV_TEST_CONFIG, 
    CONV_TUNE_CONFIG
)


def run_training():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'finetune', 'resume'], default='test')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model to use: ConvolutionalSmall, ImpalaLike, or ImpalaLarge')
    parser.add_argument('--checkpoint', type=str, default='finetune.pt', 
                       help='Checkpoint path for fine-tuning or resuming')
    parser.add_argument('--num_eval_episodes', type=int, default=9,
                       help='Number of episodes to run during evaluation')
    args = parser.parse_args()
    
    model_map = {
        'ConvolutionalSmall': ConvolutionalSmall,
        'ImpalaLike': ImpalaLike,
        'ImpalaLarge': ImpalaLarge,
    }
    
    if args.model is None:
        while True:
            model_choice = input("Select model (ConvolutionalSmall/ImpalaLike/ImpalaLarge/exit): ").strip()
            if model_choice.lower() == 'exit':
                print("Exiting program.")
                return
            elif model_choice in model_map:
                model = model_map[model_choice]
                break
            else:
                print(f"Unrecognized model '{model_choice}'. Valid options: {list(model_map.keys())}")
    else:
        if args.model not in model_map:
            print(f"Unrecognized model '{args.model}'. Valid options: {list(model_map.keys())}")
            return
        model = model_map[args.model]
    
    config_map = {
        (ImpalaLike, 'train'): IMPALA_TRAIN_CONFIG,
        (ImpalaLike, 'test'): IMPALA_TEST_CONFIG,
        (ImpalaLike, 'finetune'): IMPALA_TUNE_CONFIG,
        (ImpalaLike, 'resume'): IMPALA_TRAIN_CONFIG,
        (ImpalaLarge, 'train'): IMPALA_LARGE_TRAIN_CONFIG,
        (ImpalaLarge, 'test'): IMPALA_LARGE_TEST_CONFIG,
        (ImpalaLarge, 'finetune'): IMPALA_LARGE_TUNE_CONFIG,
        (ImpalaLarge, 'resume'): IMPALA_LARGE_TRAIN_CONFIG,
        (ConvolutionalSmall, 'train'): CONV_TRAIN_CONFIG,
        (ConvolutionalSmall, 'test'): CONV_TEST_CONFIG,
        (ConvolutionalSmall, 'finetune'): CONV_TUNE_CONFIG,
        (ConvolutionalSmall, 'resume'): CONV_TRAIN_CONFIG,
    }
    
    config = config_map[(model, args.mode)]
    
    from train import train, finetune, resume
    
    if args.mode == 'finetune':
        finetune(model, args.checkpoint, config, args.num_eval_episodes)
    elif args.mode == 'resume':
        resume(model, args.checkpoint, config, args.num_eval_episodes)
    else:
        train(model, config, args.num_eval_episodes)
