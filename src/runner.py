import argparse
from dataclasses import replace
from models import ConvolutionalSmall, ImpalaLike, TransPala
from config import (
    IMPALA_TRAIN_CONFIG, IMPALA_TEST_CONFIG, IMPALA_TUNE_CONFIG,
    TRANSPALA_TRAIN_CONFIG, TRANSPALA_TEST_CONFIG, TRANSPALA_TUNE_CONFIG,
    CONV_TRAIN_CONFIG, CONV_TEST_CONFIG, CONV_TUNE_CONFIG,
)

# Model registry: name -> (class, train_config, test_config, tune_config)
MODELS = {
    'ConvolutionalSmall': (ConvolutionalSmall, CONV_TRAIN_CONFIG, CONV_TEST_CONFIG, CONV_TUNE_CONFIG),
    'ImpalaLike': (ImpalaLike, IMPALA_TRAIN_CONFIG, IMPALA_TEST_CONFIG, IMPALA_TUNE_CONFIG),
    'TransPala': (TransPala, TRANSPALA_TRAIN_CONFIG, TRANSPALA_TEST_CONFIG, TRANSPALA_TUNE_CONFIG),
}

# Shortcut aliases
MODELS['1'] = MODELS['ConvolutionalSmall']
MODELS['2'] = MODELS['ImpalaLike']
MODELS['3'] = MODELS['TransPala']


def prompt_choice(prompt, valid_options, allow_exit=True):
    """Generic prompt helper. Returns None if user exits."""
    while True:
        choice = input(prompt).strip()
        if allow_exit and choice.lower() == 'exit':
            return None
        if choice in valid_options:
            return choice
        print(f"Invalid choice. Options: {list(valid_options)}")


def select_model():
    """Prompt user to select a model."""
    print("\n" + "="*50)
    print("MODEL SELECTION")
    print("="*50)
    for i, name in enumerate(['ConvolutionalSmall', 'ImpalaLike', 'TransPala'], 1):
        print(f"  {i}. {name}")
    
    choice = prompt_choice("Select (1/2/3 or name, 'exit' to quit): ", MODELS.keys())
    return MODELS[choice][0] if choice else None


def select_curriculum():
    """Prompt user to select curriculum option."""
    from curriculum import SCHEDULES, Curriculum
    
    print("\n" + "="*50)
    print("CURRICULUM OPTIONS")
    print("="*50)
    
    for num, schedule in SCHEDULES.items():
        print(f"\nCurriculum {num}:")
        curr = Curriculum(schedule=schedule)
        for i, (end, _) in enumerate(schedule):
            start_pct = int(schedule[i-1][0] * 100) if i > 0 else 0
            print(f"  {start_pct}-{int(end*100)}%: {curr.describe(i).split(': ', 1)[1]}")
        print(f"  Levels: {', '.join(sorted(curr.trained_levels))}")
    
    choice = prompt_choice(f"\nSelect curriculum ({list(SCHEDULES.keys())}): ", 
                          {str(k) for k in SCHEDULES.keys()}, allow_exit=False)
    return int(choice)


def print_checkpoint_info(path):
    """Display checkpoint info, return info dict."""
    from utils import get_checkpoint_info
    
    info = get_checkpoint_info(path)
    print(f"\n{'='*50}")
    print("CHECKPOINT INFO")
    print('='*50)
    print(f"  Architecture: {info['architecture']}")
    print(f"  Step: {info['step']:,} | Episode: {info['episode_num']}")
    
    if info['total_steps']:
        progress = info['step'] / info['total_steps'] * 100
        print(f"  Progress: {progress:.1f}% of {info['total_steps']:,}")
    
    print(f"  Curriculum: {'Enabled' if info['use_curriculum'] else 'Disabled'}")
    if info['curriculum_option']:
        print(f"  Curriculum option: {info['curriculum_option']}")
    
    return info


def get_config(model_class, mode):
    """Get config for model and mode."""
    _, train, test, tune = MODELS[model_class.__name__]
    return {'train': train, 'test': test, 'finetune': tune}[mode]


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'finetune', 'resume'], default='test')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='finetune.pt')
    parser.add_argument('--num_eval_episodes', type=int, default=9)
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--curriculum_option', type=int, default=None)
    parser.add_argument('--total_steps', type=int, default=None)
    args = parser.parse_args()
    
    # Model selection
    if args.model is None:
        model_class = select_model()
        if model_class is None:
            print("Exiting.")
            return
    elif args.model not in MODELS:
        print(f"Unknown model '{args.model}'. Options: 1, 2, 3 or {list(MODELS.keys())[:3]}")
        return
    else:
        model_class = MODELS[args.model][0]
    
    # Determine curriculum option
    curriculum_option = None
    
    if args.mode == 'resume':
        # Load checkpoint info and validate
        ckpt_info = print_checkpoint_info(args.checkpoint)
        
        if ckpt_info['architecture'] and ckpt_info['architecture'] != model_class.__name__:
            print(f"\n  WARNING: Checkpoint is {ckpt_info['architecture']}, selected {model_class.__name__}")
            if input("  Continue? [y/N]: ").strip().lower() != 'y':
                return
        
        # Inherit curriculum from checkpoint or args
        if args.curriculum or ckpt_info['use_curriculum']:
            curriculum_option = (args.curriculum_option or 
                               ckpt_info['curriculum_option'] or 
                               select_curriculum())
        
        if input(f"\n{'='*50}\nResume training? [y/N]: ").strip().lower() != 'y':
            return
            
        config = get_config(model_class, 'train')
        
    else:
        # Non-resume modes
        if args.curriculum:
            from curriculum import SCHEDULES
            if args.curriculum_option and args.curriculum_option in SCHEDULES:
                curriculum_option = args.curriculum_option
            else:
                curriculum_option = select_curriculum()
        
        config = get_config(model_class, args.mode)
    
    # Apply config overrides
    if curriculum_option:
        config = replace(config, use_curriculum=True)
    if args.total_steps:
        config = replace(config, num_training_steps=args.total_steps)
    
    # Run training
    from train import train
    
    train(
        model_class, 
        config, 
        num_eval_episodes=args.num_eval_episodes,
        curriculum_option=curriculum_option,
        checkpoint_path=args.checkpoint if args.mode in ('resume', 'finetune') else None,
        resume=(args.mode == 'resume'),
    )
