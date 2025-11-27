import argparse
from models import ConvolutionalSmall, ImpalaLike, TransPala
from config import (
    IMPALA_TRAIN_CONFIG, 
    IMPALA_TEST_CONFIG, 
    IMPALA_TUNE_CONFIG,
    TRANSPALA_TRAIN_CONFIG,
    TRANSPALA_TEST_CONFIG,
    TRANSPALA_TUNE_CONFIG,
    CONV_TRAIN_CONFIG, 
    CONV_TEST_CONFIG, 
    CONV_TUNE_CONFIG
)

# Model registry with numeric shortcuts
MODEL_MAP = {
    'ConvolutionalSmall': ConvolutionalSmall,
    'ImpalaLike': ImpalaLike,
    'TransPala': TransPala,
    # Numeric shortcuts
    '1': ConvolutionalSmall,
    '2': ImpalaLike,
    '3': TransPala,
}

MODEL_NAMES = ['ConvolutionalSmall', 'ImpalaLike', 'TransPala']


def select_curriculum() -> int:
    """Prompt user to select a curriculum option."""
    from curriculum import CURRICULUM_OPTIONS, CurriculumState
    
    print("\n" + "="*60)
    print("CURRICULUM OPTIONS")
    print("="*60)
    
    for option_num, schedule in CURRICULUM_OPTIONS.items():
        print(f"\nCurriculum {option_num}:")
        temp_state = CurriculumState(schedule=schedule)
        for phase_idx in range(len(schedule)):
            end_pct = int(schedule[phase_idx][0] * 100)
            start_pct = int(schedule[phase_idx - 1][0] * 100) if phase_idx > 0 else 0
            print(f"  {start_pct}-{end_pct}%: {temp_state.get_description(phase_idx).replace(f'Phase {phase_idx}: ', '')}")
        
        # Show trained levels
        trained = temp_state.get_all_trained_levels()
        print(f"  Trained levels: {', '.join(sorted(trained))}")
    
    print("\n" + "="*60)
    
    while True:
        choice = input(f"Select curriculum ({', '.join(map(str, CURRICULUM_OPTIONS.keys()))}): ").strip()
        try:
            choice_int = int(choice)
            if choice_int in CURRICULUM_OPTIONS:
                return choice_int
            else:
                print(f"Invalid choice. Please select from: {list(CURRICULUM_OPTIONS.keys())}")
        except ValueError:
            print(f"Invalid input. Please enter a number: {list(CURRICULUM_OPTIONS.keys())}")


def select_model():
    """Prompt user to select a model with numeric shortcuts."""
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    for i, name in enumerate(MODEL_NAMES, 1):
        print(f"  {i}. {name}")
    print("="*60)
    
    while True:
        choice = input("Select model (1/2/3 or name, 'exit' to quit): ").strip()
        if choice.lower() == 'exit':
            return None
        if choice in MODEL_MAP:
            return MODEL_MAP[choice]
        print(f"Invalid choice. Enter 1-3 or full model name.")


def show_checkpoint_info(checkpoint_path):
    """Display information about a checkpoint."""
    from utils import get_checkpoint_info
    from curriculum import CURRICULUM_OPTIONS, CurriculumState
    
    info = get_checkpoint_info(checkpoint_path)
    
    print(f"\n{'='*60}")
    print("CHECKPOINT INFO")
    print('='*60)
    print(f"  Architecture: {info['architecture']}")
    print(f"  Current step: {info['step']:,}")
    print(f"  Episode: {info['episode_num']}")
    if info['total_steps']:
        print(f"  Total steps: {info['total_steps']:,}")
        progress = (info['step'] / info['total_steps']) * 100
        print(f"  Progress: {progress:.1f}%")
    
    if info['use_curriculum']:
        print(f"\n  Curriculum: Enabled")
        if info['curriculum_option']:
            print(f"  Curriculum option: {info['curriculum_option']}")
    else:
        print(f"\n  Curriculum: Disabled")
    
    return info


def show_curriculum_progress(curriculum_option, current_step, total_steps):
    """Display curriculum phases and highlight current one."""
    from curriculum import CURRICULUM_OPTIONS
    
    schedule = CURRICULUM_OPTIONS[curriculum_option]
    
    print(f"\n  Curriculum {curriculum_option} phases:")
    for i, (end_progress, weights) in enumerate(schedule):
        start_pct = int(schedule[i-1][0] * 100) if i > 0 else 0
        end_pct = int(end_progress * 100)
        
        total_w = sum(weights.values())
        level_str = ", ".join(f"{int(w/total_w*100)}% {lvl}" for lvl, w in weights.items())
        
        progress = current_step / total_steps if total_steps > 0 else 0
        prev_end = schedule[i-1][0] if i > 0 else 0
        is_current = prev_end <= progress < end_progress
        
        marker = " <-- YOU ARE HERE" if is_current else ""
        print(f"    Phase {i} ({start_pct}-{end_pct}%): {level_str}{marker}")


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'finetune', 'resume'], default='test')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model: 1/ConvolutionalSmall, 2/ImpalaLike, 3/TransPala')
    parser.add_argument('--checkpoint', type=str, default='finetune.pt', 
                       help='Checkpoint path for fine-tuning or resuming')
    parser.add_argument('--num_eval_episodes', type=int, default=9,
                       help='Number of episodes to run during evaluation')
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning')
    parser.add_argument('--curriculum_option', type=int, default=None,
                       help='Curriculum option (1 or 2). If not provided, will prompt or detect from checkpoint.')
    parser.add_argument('--total_steps', type=int, default=None,
                       help='Override total training steps (useful for extending training)')
    args = parser.parse_args()
    
    # Handle model selection
    if args.model is None:
        model = select_model()
        if model is None:
            print("Exiting program.")
            return
    else:
        if args.model not in MODEL_MAP:
            print(f"Unrecognized model '{args.model}'. Valid options: 1, 2, 3 or {MODEL_NAMES}")
            return
        model = MODEL_MAP[args.model]
    
    # Get model name for config lookup
    model_name = model.__name__
    
    # Handle resume mode specially - detect curriculum from checkpoint
    if args.mode == 'resume':
        from utils import get_checkpoint_info
        from curriculum import CURRICULUM_OPTIONS
        
        # Show checkpoint info
        ckpt_info = show_checkpoint_info(args.checkpoint)
        
        # Validate architecture matches
        if ckpt_info['architecture'] and ckpt_info['architecture'] != model_name:
            print(f"\n  WARNING: Checkpoint architecture ({ckpt_info['architecture']}) "
                  f"doesn't match selected model ({model_name})")
            confirm = input("  Continue anyway? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("Aborted.")
                return
        
        # Determine curriculum option for resume
        curriculum_option = None
        if args.curriculum or ckpt_info['use_curriculum']:
            # Curriculum was enabled - determine which option
            if args.curriculum_option is not None:
                curriculum_option = args.curriculum_option
            elif ckpt_info['curriculum_option'] is not None:
                curriculum_option = ckpt_info['curriculum_option']
                print(f"\n  Detected curriculum option {curriculum_option} from checkpoint")
            else:
                print("\n  Curriculum was enabled but option not stored in checkpoint.")
                curriculum_option = select_curriculum()
            
            # Show curriculum progress
            total_steps = args.total_steps or ckpt_info['total_steps'] or 4_000_000
            show_curriculum_progress(curriculum_option, ckpt_info['step'], total_steps)
        
        # Confirm resume
        print(f"\n{'='*60}")
        confirm = input("Resume training? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return
        
        # Build config
        config_map = {
            'ImpalaLike': IMPALA_TRAIN_CONFIG,
            'TransPala': TRANSPALA_TRAIN_CONFIG,
            'ConvolutionalSmall': CONV_TRAIN_CONFIG,
        }
        config = config_map[model_name]
        
        # Apply overrides
        from dataclasses import replace
        if curriculum_option is not None:
            config = replace(config, use_curriculum=True)
        if args.total_steps is not None:
            config = replace(config, num_training_steps=args.total_steps)
        
        from train import resume
        resume(model, args.checkpoint, config, args.num_eval_episodes, curriculum_option=curriculum_option)
        return
    
    # Handle curriculum selection for non-resume modes
    curriculum_option = None
    if args.curriculum:
        from curriculum import CURRICULUM_OPTIONS
        if args.curriculum_option is not None:
            if args.curriculum_option in CURRICULUM_OPTIONS:
                curriculum_option = args.curriculum_option
            else:
                print(f"Invalid curriculum option: {args.curriculum_option}")
                print(f"Valid options: {list(CURRICULUM_OPTIONS.keys())}")
                return
        else:
            curriculum_option = select_curriculum()
    
    # Build config for other modes
    config_map = {
        # ImpalaLike mappings
        (ImpalaLike, 'train'): IMPALA_TRAIN_CONFIG,
        (ImpalaLike, 'test'): IMPALA_TEST_CONFIG,
        (ImpalaLike, 'finetune'): IMPALA_TUNE_CONFIG,
        
        # TransPala mappings
        (TransPala, 'train'): TRANSPALA_TRAIN_CONFIG,
        (TransPala, 'test'): TRANSPALA_TEST_CONFIG,
        (TransPala, 'finetune'): TRANSPALA_TUNE_CONFIG,
        
        # ConvolutionalSmall mappings
        (ConvolutionalSmall, 'train'): CONV_TRAIN_CONFIG,
        (ConvolutionalSmall, 'test'): CONV_TEST_CONFIG,
        (ConvolutionalSmall, 'finetune'): CONV_TUNE_CONFIG,
    }
    
    config = config_map[(model, args.mode)]
    
    # Enable curriculum if requested
    if curriculum_option is not None:
        from dataclasses import replace
        config = replace(config, use_curriculum=True)
    
    # Apply total_steps override
    if args.total_steps is not None:
        from dataclasses import replace
        config = replace(config, num_training_steps=args.total_steps)

    from train import train, finetune
    
    if args.mode == 'finetune':
        finetune(model, args.checkpoint, config, args.num_eval_episodes, curriculum_option=curriculum_option)
    else:
        # train and test modes
        train(model, config, args.num_eval_episodes, curriculum_option=curriculum_option)
