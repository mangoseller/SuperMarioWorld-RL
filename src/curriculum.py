"""
Curriculum learning for progressive training difficulty.

Phases are defined as (end_progress, level_weights) tuples where:
- end_progress: Training progress (0-1) at which this phase ends (based on training steps)
- level_weights: Dict mapping level names to their proportion of environments

The weights don't need to sum to exactly 1.0 - they're normalized automatically.
This allows intuitive specifications like {A: 1, B: 1} for 50/50 split.
"""

from dataclasses import dataclass
import random


# All available Super Mario World levels for evaluation
ALL_LEVELS = [
    'Bridges1', 'Bridges2',
    'ChocolateIsland1', 'ChocolateIsland2', 'ChocolateIsland3',
    'DonutPlains1', 'DonutPlains2', 'DonutPlains3', 'DonutPlains4', 'DonutPlains5',
    'Forest1', 'Forest2', 'Forest3', 'Forest4', 'Forest5',
    'VanillaDome1', 'VanillaDome2', 'VanillaDome3', 'VanillaDome4', 'VanillaDome5',
    'YoshiIsland1', 'YoshiIsland2', 'YoshiIsland3', 'YoshiIsland4',
]


# Curriculum 1: Progressive difficulty with DonutPlains focus
# - First 20%: Easy level only (YoshiIsland2)
# - Next 20%: Introduce DonutPlains1
# - Next 20%: Add DonutPlains3
# - Final 40%: Full mix with DonutPlains4
CURRICULUM_1 = [
    (0.05, {'YoshiIsland2': 1.0}),
    (0.25, {'YoshiIsland2': 0.3, 'DonutPlains1': 0.7}),
    (0.30, {'YoshiIsland2': 0.1, 'DonutPlains1': 0.3, 'DonutPlains4': 0.6}),
    (1.00, {'YoshiIsland2': 0.1, 'DonutPlains1': 0.1, 'DonutPlains4': 0.4, 'DonutPlains5': 0.4}),
]

# Curriculum 2: Slower progression, ends with YoshiIsland3
# - First 30%: Easy level only (YoshiIsland2)  
# - Next 30%: Mix YoshiIsland2 and DonutPlains1
# - Final 40%: Add YoshiIsland3 as primary
CURRICULUM_2 = [
    (0.30, {'YoshiIsland2': 1.0}),
    (0.60, {'YoshiIsland2': 0.4, 'DonutPlains1': 0.6}),
    (1.00, {'YoshiIsland2': 0.1, 'DonutPlains1': 0.3, 'YoshiIsland3': 0.6}),
]

# Available curriculum options
CURRICULUM_OPTIONS = {
    1: CURRICULUM_1,
    2: CURRICULUM_2,
}

# Default curriculum (for backwards compatibility)
DEFAULT_CURRICULUM = CURRICULUM_1


@dataclass
class CurriculumState:
    """Tracks curriculum progress during training."""
    schedule: list
    current_phase: int = 0
    
    def get_phase_weights(self, phase_idx: int = None) -> dict:
        """Get the level weights for a given phase (or current phase)."""
        idx = phase_idx if phase_idx is not None else self.current_phase
        idx = min(idx, len(self.schedule) - 1)
        return self.schedule[idx][1]
    
    def get_phase_end(self, phase_idx: int = None) -> float:
        """Get the end progress for a given phase."""
        idx = phase_idx if phase_idx is not None else self.current_phase
        idx = min(idx, len(self.schedule) - 1)
        return self.schedule[idx][0]
    
    def get_all_trained_levels(self) -> set:
        """Get all unique levels that appear in any phase of this curriculum."""
        levels = set()
        for _, weights in self.schedule:
            levels.update(weights.keys())
        return levels
    
    def get_eval_levels(self) -> list:
        """
        Get levels for evaluation: all trained levels plus one random holdout level.
        
        Returns a list of level names for evaluation.
        """
        trained_levels = self.get_all_trained_levels()
        
        # Find holdout levels (levels not in curriculum)
        holdout_candidates = [level for level in ALL_LEVELS if level not in trained_levels]
        
        eval_levels = list(trained_levels)
        
        # Add one random holdout level if available
        if holdout_candidates:
            random_holdout = random.choice(holdout_candidates)
            eval_levels.append(random_holdout)
        
        return sorted(eval_levels)
    
    def check_phase_transition(self, step: int, total_steps: int) -> bool:
        """
        Check if we should transition to a new phase.
        
        Args:
            step: Current training step (iteration of the main training loop)
            total_steps: Total number of training steps
        
        Returns True if phase changed, False otherwise.
        Updates current_phase internally.
        """
        if total_steps == 0:
            return False
            
        progress = step / total_steps
        new_phase = self._get_phase_for_progress(progress)
        
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            return True
        return False
    
    def _get_phase_for_progress(self, progress: float) -> int:
        """Determine which phase index corresponds to given progress."""
        for i, (end_progress, _) in enumerate(self.schedule):
            if progress < end_progress:
                return i
        return len(self.schedule) - 1
    
    def get_description(self, phase_idx: int = None) -> str:
        """Get human-readable description of a phase."""
        idx = phase_idx if phase_idx is not None else self.current_phase
        weights = self.get_phase_weights(idx)
        total = sum(weights.values())
        
        parts = [f"{int(w/total*100)}% {level}" for level, w in weights.items()]
        return f"Phase {idx}: " + ", ".join(parts)
    
    @classmethod
    def from_schedule(cls, schedule: list = None):
        """Create a CurriculumState from a schedule (or use default)."""
        return cls(schedule=schedule or DEFAULT_CURRICULUM)
    
    @classmethod
    def from_option(cls, option: int):
        """Create a CurriculumState from a curriculum option number (1 or 2)."""
        if option not in CURRICULUM_OPTIONS:
            raise ValueError(f"Invalid curriculum option: {option}. Choose from {list(CURRICULUM_OPTIONS.keys())}")
        return cls(schedule=CURRICULUM_OPTIONS[option])


def compute_level_distribution(num_envs: int, level_weights: dict) -> list:
    """
    Distribute environments across levels according to weights.
    
    Args:
        num_envs: Total number of environments to distribute
        level_weights: Dict mapping level names to relative weights.
                      Weights are normalized, so {A: 1, B: 1} gives 50/50,
                      and {A: 0.7, B: 0.3} also works as expected.
    
    Returns:
        List of level names with length == num_envs
    
    Example:
        >>> compute_level_distribution(10, {'LevelA': 0.7, 'LevelB': 0.3})
        ['LevelA', 'LevelA', 'LevelA', 'LevelA', 'LevelA', 'LevelA', 'LevelA', 
         'LevelB', 'LevelB', 'LevelB']
    """
    if not level_weights:
        raise ValueError("level_weights cannot be empty")
    
    # Normalize weights
    total_weight = sum(level_weights.values())
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")
    
    normalized = {k: v / total_weight for k, v in level_weights.items()}
    
    # Distribute environments using largest remainder method for fairness
    levels = list(normalized.keys())
    exact_counts = {level: normalized[level] * num_envs for level in levels}
    floor_counts = {level: int(exact_counts[level]) for level in levels}
    remainders = {level: exact_counts[level] - floor_counts[level] for level in levels}
    
    # Start with floor counts
    distribution = []
    for level in levels:
        distribution.extend([level] * floor_counts[level])
    
    # Distribute remaining slots to levels with largest remainders
    remaining = num_envs - len(distribution)
    sorted_by_remainder = sorted(levels, key=lambda l: remainders[l], reverse=True)
    
    for i in range(remaining):
        distribution.append(sorted_by_remainder[i % len(sorted_by_remainder)])
    
    return distribution


# Convenience function for simple uniform distribution
def uniform_distribution(num_envs: int, *levels: str) -> list:
    """
    Distribute environments uniformly across specified levels.
    
    Args:
        num_envs: Total number of environments
        *levels: Level names to distribute across
    
    Example:
        >>> uniform_distribution(10, 'LevelA', 'LevelB')
        ['LevelA', 'LevelA', 'LevelA', 'LevelA', 'LevelA',
         'LevelB', 'LevelB', 'LevelB', 'LevelB', 'LevelB']
    """
    if not levels:
        raise ValueError("At least one level must be specified")
    weights = {level: 1.0 for level in levels}
    return compute_level_distribution(num_envs, weights)
