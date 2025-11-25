import argparse
from models import ConvolutionalSmall, ImpalaLike
from config import (
    IMPALA_TRAIN_CONFIG, 
    IMPALA_TEST_CONFIG, 
    IMPALA_TUNE_CONFIG,
    CONV_TRAIN_CONFIG, 
    CONV_TEST_CONFIG, 
    CONV_TUNE_CONFIG
)


def run_training():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'finetune', 'resume'], default='test')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model to use: ConvolutionalSmall or ImpalaLike')
    parser.add_argument('--checkpoint', type=str, default='finetune.pt', 
                       help='Checkpoint path for fine-tuning or resuming')
    parser.add_argument('--num_eval_episodes', type=int, default=9,
                       help='Number of episodes to run during evaluation')
    args = parser.parse_args()
    
    # Get model selection
    if args.model is None:
        while True:
            model_choice = input("Select model (ConvolutionalSmall/ImpalaLike/exit): ").strip()
            if model_choice.lower() == 'exit':
                print("Exiting program.")
                return
            elif model_choice == 'ConvolutionalSmall':
                model = ConvolutionalSmall
                break
            elif model_choice == 'ImpalaLike':
                model = ImpalaLike
                break
            else:
                print(f"Unrecognized model '{model_choice}'. Please choose ConvolutionalSmall or ImpalaLike.")
    else:
        if args.model == 'ConvolutionalSmall':
            model = ConvolutionalSmall
        elif args.model == 'ImpalaLike':
            model = ImpalaLike
        else:
            print(f"Unrecognized model '{args.model}'. Valid options: ConvolutionalSmall, ImpalaLike")
            return
    
    config_map = {
        (ImpalaLike, 'train'): IMPALA_TRAIN_CONFIG,
        (ImpalaLike, 'test'): IMPALA_TEST_CONFIG,
        (ImpalaLike, 'finetune'): IMPALA_TUNE_CONFIG,
        (ImpalaLike, 'resume'): IMPALA_TRAIN_CONFIG,  # Resume uses train config
        (ConvolutionalSmall, 'train'): CONV_TRAIN_CONFIG,
        (ConvolutionalSmall, 'test'): CONV_TEST_CONFIG,
        (ConvolutionalSmall, 'finetune'): CONV_TUNE_CONFIG,
        (ConvolutionalSmall, 'resume'): CONV_TRAIN_CONFIG,  # Resume uses train config
    }
    
    config = config_map[(model, args.mode)]
    
    # Import train functions here to avoid circular imports
    from train import train, finetune, resume
    
    if args.mode == 'finetune':
        finetune(model, args.checkpoint, config, args.num_eval_episodes)
    elif args.mode == 'resume':
        resume(model, args.checkpoint, config, args.num_eval_episodes)
    else:
        train(model, config, args.num_eval_episodes)
