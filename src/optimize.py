"""
optimize.py - Highly optimized Simulated Annealing for Santa 2025
Key improvements over baseline:
- 5-10x more iterations
- Better temperature schedule
- Multiple move types
- Adaptive step sizes
- Multi-restart capability
"""
import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .geometry import (
    make_tree_polygon, bounding_square_side, has_collision,
    compute_bounding_square_side, center_placements, placement_in_bounds
)

@dataclass
class OptimizationConfig:
    # Iteration counts - MUCH HIGHER than baseline
    sa_iterations_base: int = 5000      # Base: 5000 (baseline was ~1500)
    
    # Temperature schedule
    temp_initial_factor: float = 1.5    # T_initial = factor * initial_side
    temp_final_ratio: float = 0.0001    # T_final = ratio * T_initial
    
    # Move parameters
    max_shift_factor: float = 0.25      # Initial shift = factor * side
    max_rotate: float = 45.0            # Initial max rotation (degrees)
    shift_decay: float = 0.9995         # Per-iteration decay
    rotate_decay: float = 0.9995
    
    # Move probabilities
    prob_translate: float = 0.5
    prob_rotate: float = 0.3
    prob_combined: float = 0.2
    
    # Multi-restart
    num_restarts: int = 1
    
    seed: int = 42
    
    @classmethod
    def quick_mode(cls):
        """Fast mode for testing."""
        return cls(sa_iterations_base=2000, num_restarts=1)
    
    @classmethod
    def standard_mode(cls):
        """Standard mode - good balance."""
        return cls(sa_iterations_base=5000, num_restarts=2)
    
    @classmethod
    def aggressive_mode(cls):
        """Aggressive optimization for better scores."""
        return cls(
            sa_iterations_base=10000,
            temp_initial_factor=2.0,
            temp_final_ratio=0.00001,
            max_shift_factor=0.3,
            max_rotate=60.0,
            shift_decay=0.9998,
            rotate_decay=0.9998,
            num_restarts=3
        )
    
    @classmethod
    def maximum_mode(cls):
        """Maximum optimization - use for final submission."""
        return cls(
            sa_iterations_base=20000,
            temp_initial_factor=2.5,
            temp_final_ratio=0.000001,
            max_shift_factor=0.35,
            max_rotate=90.0,
            shift_decay=0.9999,
            rotate_decay=0.9999,
            num_restarts=5
        )


def optimize_layout(
    num_trees: int,
    initial_positions: List[Tuple[float, float, float]],
    max_iterations: int = 5000,
    config: Optional[OptimizationConfig] = None
) -> List[Tuple[float, float, float]]:
    """
    Optimize layout using Simulated Annealing.
    
    This is a highly optimized version with:
    - Better temperature schedule
    - Adaptive step sizes
    - Multiple move types
    - Best-so-far tracking
    """
    if config is None:
        config = OptimizationConfig()
    
    if num_trees <= 1:
        return initial_positions
    
    # Initialize
    positions = list(initial_positions)
    polys = [make_tree_polygon(x, y, ang) for x, y, ang in positions]
    
    current_side = bounding_square_side(polys)
    best_side = current_side
    best_positions = positions.copy()
    
    # Temperature schedule
    T_initial = config.temp_initial_factor * current_side
    T_final = config.temp_final_ratio * T_initial
    cooling_rate = math.exp(math.log(T_final / T_initial) / max_iterations)
    
    # Adaptive step sizes
    max_shift = config.max_shift_factor * current_side
    max_rotate = config.max_rotate
    
    T = T_initial
    
    for iteration in range(max_iterations):
        # Pick random tree
        i = random.randrange(num_trees)
        x, y, ang = positions[i]
        
        # Choose move type
        r = random.random()
        
        if r < config.prob_translate:
            # Translation move
            dx = random.uniform(-max_shift, max_shift)
            dy = random.uniform(-max_shift, max_shift)
            new_x, new_y, new_ang = x + dx, y + dy, ang
            
        elif r < config.prob_translate + config.prob_rotate:
            # Rotation move
            dtheta = random.uniform(-max_rotate, max_rotate)
            new_x, new_y, new_ang = x, y, (ang + dtheta) % 360.0
            
        else:
            # Combined move
            dx = random.uniform(-max_shift * 0.7, max_shift * 0.7)
            dy = random.uniform(-max_shift * 0.7, max_shift * 0.7)
            dtheta = random.uniform(-max_rotate * 0.7, max_rotate * 0.7)
            new_x, new_y, new_ang = x + dx, y + dy, (ang + dtheta) % 360.0
        
        # Check bounds
        if not placement_in_bounds(new_x, new_y, new_ang):
            T *= cooling_rate
            max_shift *= config.shift_decay
            max_rotate *= config.rotate_decay
            continue
        
        # Create new polygon and check collision
        new_poly = make_tree_polygon(new_x, new_y, new_ang)
        others = polys[:i] + polys[i+1:]
        
        if has_collision(new_poly, others):
            T *= cooling_rate
            max_shift *= config.shift_decay
            max_rotate *= config.rotate_decay
            continue
        
        # Compute new score
        old_poly = polys[i]
        polys[i] = new_poly
        new_side = bounding_square_side(polys)
        
        # Acceptance criterion (Metropolis)
        delta = new_side - current_side
        
        if delta <= 0:
            # Improvement - always accept
            positions[i] = (new_x, new_y, new_ang)
            current_side = new_side
            
            if current_side < best_side:
                best_side = current_side
                best_positions = positions.copy()
        else:
            # Worse - accept with probability
            if T > 0 and random.random() < math.exp(-delta / T):
                positions[i] = (new_x, new_y, new_ang)
                current_side = new_side
            else:
                # Reject - revert
                polys[i] = old_poly
        
        # Update temperature and steps
        T *= cooling_rate
        max_shift = max(0.001, max_shift * config.shift_decay)
        max_rotate = max(0.1, max_rotate * config.rotate_decay)
    
    return best_positions


def optimize_with_restarts(
    num_trees: int,
    initial_positions: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    verbose: bool = False
) -> List[Tuple[float, float, float]]:
    """Run SA with multiple restarts, return best result."""
    
    best_solution = None
    best_score = float('inf')
    
    for restart in range(config.num_restarts):
        if restart == 0:
            start = list(initial_positions)
        else:
            # Perturb best solution found so far
            start = list(best_solution) if best_solution else list(initial_positions)
            for i in range(len(start)):
                if random.random() < 0.4:
                    x, y, deg = start[i]
                    start[i] = (
                        x + random.uniform(-0.15, 0.15),
                        y + random.uniform(-0.15, 0.15),
                        (deg + random.uniform(-45, 45)) % 360
                    )
        
        # Scale iterations by n for larger problems
        n_iter = int(config.sa_iterations_base * (1 + num_trees / 100))
        
        result = optimize_layout(num_trees, start, n_iter, config)
        score = compute_bounding_square_side(result)
        
        if score < best_score:
            best_score = score
            best_solution = result
            if verbose:
                print(f"    Restart {restart+1}: score={score:.4f}")
    
    return best_solution if best_solution else initial_positions


def optimize_placement(
    solution: List[Tuple[float, float, float]],
    config: Optional[OptimizationConfig] = None,
    iterations_multiplier: float = 1.0,
    verbose: bool = False
) -> List[Tuple[float, float, float]]:
    """Main optimization entry point."""
    if config is None:
        config = OptimizationConfig.standard_mode()
    
    n = len(solution)
    if n <= 1:
        return solution
    
    random.seed(config.seed + n)
    
    if config.num_restarts > 1:
        return optimize_with_restarts(n, solution, config, verbose)
    else:
        n_iter = int(config.sa_iterations_base * iterations_multiplier * (1 + n / 100))
        return optimize_layout(n, solution, n_iter, config)
