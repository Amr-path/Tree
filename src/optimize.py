"""
optimize.py - Hybrid optimizer for Santa 2025
Key components:
- Adaptive Simulated Annealing with dynamic cooling
- Differential Evolution for global exploration (optional, SciPy-backed)
- L-BFGS-B local refinement on dense cores
- Multi-resolution stages with angular sweeps for rotation discovery
- Multi-restart capability with structured perturbations
- Feasible global compaction via scaled line search + short anneal polish
"""
import math
import random
from typing import List, Tuple, Optional, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
try:  # SciPy may not be available in some execution environments
    from scipy.optimize import differential_evolution, minimize
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    differential_evolution = None
    minimize = None
    SCIPY_AVAILABLE = False

from .geometry import (
    make_tree_polygon, bounding_square_side, has_collision,
    compute_bounding_square_side, center_placements, placement_in_bounds,
    compute_bounding_box
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

    # Advanced global/local hybrid controls
    overlap_penalty: float = 25.0
    de_maxiter: int = 35
    de_popsize: int = 12
    lbfgs_maxiter: int = 45
    angular_sweep_steps: int = 16
    rotation_sweep_range: float = 90.0
    multi_resolution_levels: int = 3
    multi_resolution_shrink: float = 0.72
    max_de_trees: int = 36
    annealing_multiplier: float = 1.4
    use_differential_evolution: bool = True
    use_lbfgs: bool = True
    use_annealing: bool = True
    enable_global_compaction: bool = True
    global_scale_floor: float = 0.35
    global_scale_shrink: float = 0.82
    global_scale_tolerance: float = 1e-4
    global_scale_search_steps: int = 18
    global_compaction_iterations: float = 0.45
    
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


def _flatten_solution(solution: Sequence[Tuple[float, float, float]], indices: Optional[Iterable[int]] = None) -> np.ndarray:
    """Flatten solution into vector (x0,y0,a0,...)."""
    if indices is None:
        indices = range(len(solution))
    data = []
    for i in indices:
        x, y, ang = solution[i]
        data.extend([x, y, ang])
    return np.asarray(data, dtype=float)


def _unflatten_solution(
    base_solution: Sequence[Tuple[float, float, float]],
    vec: np.ndarray,
    indices: Optional[Iterable[int]] = None
) -> List[Tuple[float, float, float]]:
    """Apply flattened vector onto base solution positions."""
    sol = list(base_solution)
    if indices is None:
        indices = range(len(base_solution))
    idx_list = list(indices)
    for j, i in enumerate(idx_list):
        sol[i] = (float(vec[3 * j]), float(vec[3 * j + 1]), float(vec[3 * j + 2]) % 360.0)
    return sol


def _collision_penalty(polys: List) -> float:
    """Count collisions between polygons to build penalty term."""
    penalty = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if has_collision(polys[i], [polys[j]]):
                penalty += 1.0
    return penalty


def _penalized_objective(
    vec: np.ndarray,
    base_solution: Sequence[Tuple[float, float, float]],
    indices: Sequence[int],
    config: OptimizationConfig
) -> float:
    """Objective function for continuous optimizers."""
    candidate = _unflatten_solution(base_solution, vec, indices)
    polys = [make_tree_polygon(x, y, ang) for x, y, ang in candidate]
    side = bounding_square_side(polys)
    penalty = _collision_penalty(polys)
    return side + penalty * config.overlap_penalty


def _select_dense_subset(solution: Sequence[Tuple[float, float, float]], k: int) -> List[int]:
    """Pick a subset of trees closest to center to focus global search on dense core."""
    if len(solution) <= k:
        return list(range(len(solution)))
    min_x, min_y, max_x, max_y = compute_bounding_box(solution)
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    distances = [math.hypot(x - center[0], y - center[1]) for x, y, _ in solution]
    return sorted(range(len(solution)), key=lambda i: distances[i])[:k]


def _rotation_sweep(
    solution: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    level: int = 0
) -> List[Tuple[float, float, float]]:
    """Perform local angular sweeps to find interlocking rotations."""
    if not solution:
        return solution
    
    step_scale = config.multi_resolution_shrink ** level
    ang_range = config.rotation_sweep_range * step_scale
    steps = max(4, int(config.angular_sweep_steps * step_scale))
    improved = list(solution)
    polys = [make_tree_polygon(x, y, a) for x, y, a in improved]
    
    for idx, (x, y, ang) in enumerate(improved):
        best_ang = ang
        best_side = bounding_square_side(polys)
        
        for delta in np.linspace(-ang_range, ang_range, steps):
            candidate_ang = (ang + delta) % 360.0
            new_poly = make_tree_polygon(x, y, candidate_ang)
            others = polys[:idx] + polys[idx + 1:]
            if has_collision(new_poly, others):
                continue
            polys[idx] = new_poly
            side = bounding_square_side(polys)
            if side < best_side:
                best_side = side
                best_ang = candidate_ang
        polys[idx] = make_tree_polygon(x, y, best_ang)
        improved[idx] = (x, y, best_ang)
    return improved


def _run_differential_evolution(
    solution: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    level: int = 0
) -> List[Tuple[float, float, float]]:
    """Global search with Differential Evolution on a dense subset."""
    n = len(solution)
    if n == 0 or not config.use_differential_evolution or not SCIPY_AVAILABLE:
        return solution
    
    subset = _select_dense_subset(solution, min(config.max_de_trees, n))
    bounds = []
    step_scale = config.multi_resolution_shrink ** level
    limit = max(1.5, bounding_square_side([make_tree_polygon(x, y, a) for x, y, a in solution]) * 1.1)
    for _ in subset:
        bounds.extend([(-limit, limit), (-limit, limit), (0.0, 360.0)])
    
    def objective(vec):
        return _penalized_objective(vec, solution, subset, config)
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=max(10, int(config.de_maxiter * step_scale)),
        popsize=config.de_popsize,
        polish=False,
        seed=config.seed + level
    )
    return _unflatten_solution(solution, result.x, subset)


def _run_lbfgs(
    solution: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    level: int = 0
) -> List[Tuple[float, float, float]]:
    """Local refinement using L-BFGS-B on the same dense subset."""
    n = len(solution)
    if n == 0 or not config.use_lbfgs or not SCIPY_AVAILABLE:
        return solution
    
    subset = _select_dense_subset(solution, min(config.max_de_trees, n))
    bounds = []
    limit = max(1.5, bounding_square_side([make_tree_polygon(x, y, a) for x, y, a in solution]) * 1.05)
    for _ in subset:
        bounds.extend([(-limit, limit), (-limit, limit), (0.0, 360.0)])
    
    x0 = _flatten_solution(solution, subset)
    
    def objective(vec):
        return _penalized_objective(vec, solution, subset, config)
    
    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max(15, int(config.lbfgs_maxiter * (config.multi_resolution_shrink ** level)))}
    )
    
    return _unflatten_solution(solution, res.x, subset)


def _anneal_stage(
    solution: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    iterations_multiplier: float,
    level: int
) -> List[Tuple[float, float, float]]:
    """Run adaptive simulated annealing with level-scaled iterations."""
    n_iter = int(
        config.sa_iterations_base
        * iterations_multiplier
        * (1 + len(solution) / 100)
        * (config.annealing_multiplier ** level)
    )
    return optimize_layout(len(solution), solution, n_iter, config, adaptive=True)


def _hybrid_stage(
    solution: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    iterations_multiplier: float,
    level: int,
    verbose: bool = False
) -> List[Tuple[float, float, float]]:
    """Single multi-resolution stage combining sweeps, DE, LBFGS, and annealing."""
    working = center_placements(solution)
    working = _rotation_sweep(working, config, level)
    if config.use_differential_evolution:
        working = _run_differential_evolution(working, config, level)
    if config.use_lbfgs:
        working = _run_lbfgs(working, config, level)
    if config.use_annealing:
        working = _anneal_stage(working, config, iterations_multiplier, level)
    working = center_placements(working)
    
    if verbose:
        score = compute_bounding_square_side(working)
        print(f"    Level {level+1}: side={score:.4f}")
    return working


def _global_compaction_stage(
    solution: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    level: int = 0
) -> List[Tuple[float, float, float]]:
    """Feasible radial compaction followed by a short anneal polish."""
    compacted = _global_scale_search(solution, config)
    if config.use_annealing and config.global_compaction_iterations > 0:
        compacted = _anneal_stage(
            compacted,
            config,
            iterations_multiplier=config.global_compaction_iterations,
            level=level
        )
    return center_placements(compacted)


def _multi_resolution_pipeline(
    solution: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    iterations_multiplier: float = 1.0,
    verbose: bool = False
) -> List[Tuple[float, float, float]]:
    """Run multi-resolution hybrid optimization stack."""
    working = list(solution)
    for level in range(config.multi_resolution_levels):
        working = _hybrid_stage(working, config, iterations_multiplier, level, verbose=verbose)
    
    if config.enable_global_compaction:
        working = _global_compaction_stage(working, config, level=config.multi_resolution_levels)
    return working


def _perturb_solution(solution: List[Tuple[float, float, float]], magnitude: float = 0.2) -> List[Tuple[float, float, float]]:
    """Small random perturbation used for restarts."""
    perturbed = []
    for x, y, ang in solution:
        perturbed.append((
            x + random.uniform(-magnitude, magnitude),
            y + random.uniform(-magnitude, magnitude),
            (ang + random.uniform(-magnitude * 180, magnitude * 180)) % 360.0
        ))
    return perturbed


def _scaled_solution(solution: List[Tuple[float, float, float]], scale: float) -> List[Tuple[float, float, float]]:
    """Apply uniform scaling about origin."""
    return [(x * scale, y * scale, ang) for x, y, ang in solution]


def _is_feasible_scale(solution: List[Tuple[float, float, float]], scale: float) -> bool:
    """Check if uniformly scaling placements remains overlap-free."""
    if not solution:
        return True
    polys = []
    for x, y, ang in solution:
        poly = make_tree_polygon(x * scale, y * scale, ang)
        if has_collision(poly, polys):
            return False
        polys.append(poly)
    return True


def _global_scale_search(
    solution: List[Tuple[float, float, float]],
    config: OptimizationConfig
) -> List[Tuple[float, float, float]]:
    """
    Compact layout with a feasibility-preserving line search on global scale.
    
    Uses geometric progression to find an infeasible lower bound and then
    bisects to the tightest collision-free scale. This replaces heuristic
    radius shrinking with a mathematically controlled search.
    """
    if len(solution) <= 1 or not config.enable_global_compaction:
        return solution
    
    upper = 1.0
    lower = None
    scale = upper * config.global_scale_shrink
    
    while scale >= config.global_scale_floor:
        if _is_feasible_scale(solution, scale):
            upper = scale
            scale *= config.global_scale_shrink
        else:
            lower = scale
            break
    
    # If still feasible at floor, return the tightest discovered scale.
    if lower is None:
        return center_placements(_scaled_solution(solution, upper))
    
    for _ in range(config.global_scale_search_steps):
        mid = (upper + lower) / 2
        if _is_feasible_scale(solution, mid):
            upper = mid
        else:
            lower = mid
        if abs(upper - lower) < config.global_scale_tolerance:
            break
    
    return center_placements(_scaled_solution(solution, upper))


def optimize_layout(
    num_trees: int,
    initial_positions: List[Tuple[float, float, float]],
    max_iterations: int = 5000,
    config: Optional[OptimizationConfig] = None,
    adaptive: bool = False
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
    accept_count = 0
    attempt_count = 0
    
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
            accept_count += 1
            
            if current_side < best_side:
                best_side = current_side
                best_positions = positions.copy()
        else:
            # Worse - accept with probability
            if T > 0 and random.random() < math.exp(-delta / T):
                positions[i] = (new_x, new_y, new_ang)
                current_side = new_side
                accept_count += 1
            else:
                # Reject - revert
                polys[i] = old_poly
        attempt_count += 1
        
        # Update temperature and steps
        T *= cooling_rate
        max_shift = max(0.001, max_shift * config.shift_decay)
        max_rotate = max(0.1, max_rotate * config.rotate_decay)
        
        if adaptive and (iteration + 1) % 200 == 0 and attempt_count > 0:
            acceptance_ratio = accept_count / attempt_count
            if acceptance_ratio < 0.15:
                T *= 0.85
                max_shift *= 0.9
                max_rotate *= 0.9
            elif acceptance_ratio > 0.65:
                T *= 1.15
                max_shift *= 1.05
                max_rotate *= 1.05
            accept_count = 0
            attempt_count = 0
    
    return best_positions


def optimize_with_restarts(
    num_trees: int,
    initial_positions: List[Tuple[float, float, float]],
    config: OptimizationConfig,
    verbose: bool = False,
    iterations_multiplier: float = 1.0
) -> List[Tuple[float, float, float]]:
    """Run hybrid optimizer with multiple restarts, return best result."""
    
    best_solution = None
    best_score = float('inf')
    
    for restart in range(config.num_restarts):
        if restart == 0:
            start = center_placements(initial_positions)
        else:
            start = _perturb_solution(best_solution if best_solution else initial_positions, magnitude=0.25)
        
        result = _multi_resolution_pipeline(start, config, iterations_multiplier, verbose=False)
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
    """Main optimization entry point using hybrid multi-resolution pipeline."""
    if config is None:
        config = OptimizationConfig.standard_mode()
    
    n = len(solution)
    if n <= 1:
        return solution
    
    random.seed(config.seed + n)
    
    centered = center_placements(solution)
    
    if config.num_restarts > 1:
        return optimize_with_restarts(n, centered, config, verbose, iterations_multiplier)
    
    return _multi_resolution_pipeline(centered, config, iterations_multiplier, verbose)
