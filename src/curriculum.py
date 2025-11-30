from dataclasses import dataclass
import random

@dataclass
class Curriculum:
    schedule: list
    stage: int = 0
    
    @property
    def weights(self):
        return self.schedule[min(self.stage, len(self.schedule) - 1)][1]
    
    @property
    def trained_levels(self):
        return {level for _, weights in self.schedule for level in weights}
    
    @property
    def eval_levels(self):
    # Get all training levels and a random unseen level for training
        trained = self.trained_levels
        holdouts = [level for level in ALL_LEVELS if level not in trained]
        result = list(trained)
        result.append(random.choice(holdouts))
        return sorted(result)
    
    def update(self, step, total_steps):
    # Advance to next stage if needed
        if total_steps == 0:
            return False
        
        progress = step / total_steps
        new_stage = next(
            (i for i, (end, _) in enumerate(self.schedule) if progress < end),
            len(self.schedule) - 1
        ) # Give the first item we haven't progressed past, or the final item
        
        if new_stage != self.stage:
            self.stage = new_stage
            return True
        return False
    
    def describe(self, stage):
        idx = stage if stage is not None else self.stage
        weights = self.schedule[min(idx, len(self.schedule) - 1)][1]
        total = sum(weights.values())
        parts = [f"{int(w / total * 100)}% {level}" for level, w in weights.items()]
        return f"Stage {idx}: " + ", ".join(parts)
    
    @classmethod
    def create(cls, option: int = 1):
        # Create a curriculum from a schedule 
        if option not in SCHEDULES:
            raise ValueError(f"Invalid option {option}. Choose from {list(SCHEDULES.keys())}")
        return cls(schedule=SCHEDULES[option])


def assign_levels(num_envs: int, level_weights: dict) -> list:
    
    """Distribute levels across environments according to level_weights,
    Uses largest-remainder method for allocation when counts don't divide evenly.
    Weights are normalized i.e {A: 1, B: 1} gives 50/50 split"""
    
    if not level_weights:
        raise ValueError("level_weights cannot be empty")
    levels = list(level_weights.keys())
    counts = {level: (level_weights[level] / sum(level_weights.values())) * num_envs for level in levels}
    
    result = []     
    for level in levels:
        result.extend([level] * int(counts[level]))
    
    # Distribute remaining slots to levels with largest fractional parts
    remainders = {level: counts[level] - int(counts[level]) for level in levels}
    by_remainder = sorted(levels, key=lambda l: remainders[l], reverse=True)
    
    for i in range(num_envs - len(result)):
        result.append(by_remainder[i % len(by_remainder)])
    
    return result

ALL_LEVELS = [
    'Bridges1', 'Bridges2',
    'ChocolateIsland1', 'ChocolateIsland2', 'ChocolateIsland3',
    'DonutPlains1', 'DonutPlains2', 'DonutPlains3', 'DonutPlains4', 'DonutPlains5',
    'Forest1', 'Forest2', 'Forest3', 'Forest4', 'Forest5',
    'VanillaDome1', 'VanillaDome2', 'VanillaDome3', 'VanillaDome4', 'VanillaDome5',
    'YoshiIsland1', 'YoshiIsland2', 'YoshiIsland3', 'YoshiIsland4',
]


PROGRESSIVE_SCHEDULE = [
    (0.15, {'YoshiIsland2': 0.5, 'VanillaDome5': 0.5}),
    (0.25, {'YoshiIsland2': 0.2, 'VanillaDome5': 0.8}),
    (0.30, {'YoshiIsland2': 0.1, 'VanillaDome5': 0.9}),
    (0.40, {'YoshiIsland2': 0.1, 'VanillaDome5': 0.65, 'ChocolateIsland3': 0.25}),
    (0.60, {'YoshiIsland2': 0.1, 'VanillaDome5': 0.4, 'ChocolateIsland3': 0.5}),
    (0.75, {'YoshiIsland2': 0.075, 'VanillaDome5': 0.275, 'ChocolateIsland3': 0.45, 'YoshiIsland3': 0.15}),
    (1.00, {'YoshiIsland2': 0.05, 'ChocolateIsland3': 0.4, 'VanillaDome5': 0.15, 'YoshiIsland3': 0.3}),
]

GRADUAL_SCHEDULE = [
    (0.30, {'YoshiIsland2': 1.0}),
    (0.60, {'YoshiIsland2': 0.4, 'DonutPlains1': 0.6}),
    (1.00, {'YoshiIsland2': 0.1, 'DonutPlains1': 0.3, 'YoshiIsland3': 0.6}),
]

SCHEDULES = {
    1: PROGRESSIVE_SCHEDULE,
    2: GRADUAL_SCHEDULE,
}

