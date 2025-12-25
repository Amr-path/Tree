"""
packing.py - Main packing solver for Santa 2025

Key strategies:
1. Incremental building (n=1 -> n=200)
2. Multiple initialization patterns
3. Smart tree insertion
4. Periodic reshuffling to escape local minima
"""
import math
import random
from typing import List, Tuple, Dict, Optional

from .geometry import (
    make_tree_polygon, bounding_square_side, has_collision,
    compute_bounding_square_side, compute_bounding_box,
    center_placements, check_all_overlaps, TREE_WIDTH, TREE_HEIGHT
)
from .optimize import (
    optimize_layout, optimize_placement, optimize_with_restarts,
    OptimizationConfig
)

Placement = Tuple[float, float, float]
Solution = List[Placement]


def initial_grid_positions(n: int) -> Solution:
    """Create initial grid layout (baseline approach)."""
    positions = []
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    x_spacing = 0.72  # slightly > tree width
    y_spacing = 1.02  # slightly > tree height
    
    count = 0
    for r in range(rows):
        if count >= n:
            break
        for c in range(cols):
            if count >= n:
                break
            x = c * x_spacing
            y = -r * y_spacing
            positions.append((x, y, 0.0))
            count += 1
    
    return center_placements(positions)


def initial_hexagonal_positions(n: int) -> Solution:
    """Create hexagonal/staggered grid layout for better packing."""
    positions = []
    cols = math.ceil(math.sqrt(n * 1.15))
    
    x_spacing = 0.68
    y_spacing = 0.92
    
    count = 0
    row = 0
    
    while count < n:
        # Offset alternating rows
        x_offset = (row % 2) * (x_spacing * 0.5)
        
        for c in range(cols + 1):
            if count >= n:
                break
            x = c * x_spacing + x_offset
            y = -row * y_spacing
            # Alternate rotation pattern
            deg = 180.0 if (row + c) % 2 == 1 else 0.0
            positions.append((x, y, deg))
            count += 1
        
        row += 1
    
    return center_placements(positions)


def initial_brick_positions(n: int) -> Solution:
    """Create brick-like interlocking pattern."""
    positions = []
    
    # Tighter spacing for brick pattern
    x_spacing = 0.65
    y_spacing = 0.88
    
    count = 0
    row = 0
    
    while count < n:
        x_offset = (row % 2) * (x_spacing * 0.5)
        cols_this_row = math.ceil(math.sqrt(n)) + 1
        
        for c in range(cols_this_row):
            if count >= n:
                break
            x = c * x_spacing + x_offset
            y = -row * y_spacing
            # Brick pattern: alternate 0 and 180
            deg = 0.0 if row % 2 == 0 else 180.0
            positions.append((x, y, deg))
            count += 1
        
        row += 1
    
    return center_placements(positions)


def initial_spiral_positions(n: int) -> Solution:
    """Create outward spiral pattern."""
    if n == 0:
        return []
    
    positions = [(0.0, 0.0, 0.0)]
    if n == 1:
        return positions
    
    base_spacing = 0.75
    angle = 0.0
    radius = base_spacing
    golden_angle = 2.399963  # Golden angle ~137.5 degrees
    
    for i in range(1, n):
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        deg = 180.0 if i % 2 == 1 else 0.0
        
        positions.append((x, y, deg))
        angle += golden_angle
        radius += base_spacing * 0.22 / (1 + i * 0.015)
    
    return center_placements(positions)


def find_best_insertion_position(
    existing: Solution,
    existing_polys: List,
    rotations: List[float] = [0, 45, 90, 135, 180, 225, 270, 315]
) -> Placement:
    """Find best position to insert a new tree."""
    if not existing:
        return (0.0, 0.0, 0.0)
    
    min_x, min_y, max_x, max_y = compute_bounding_box(existing)
    margin = 0.6
    
    best_placement = None
    best_score = float('inf')
    
    # Grid search around perimeter and inside
    grid_size = 12
    
    for gx in [min_x - margin] + list(range(int(min_x), int(max_x + 1))) + [max_x + margin]:
        for gy in [min_y - margin] + list(range(int(min_y), int(max_y + 1))) + [max_y + margin]:
            # Convert to float
            gx_f = float(gx) if isinstance(gx, int) else gx
            gy_f = float(gy) if isinstance(gy, int) else gy
            
            for deg in rotations:
                poly = make_tree_polygon(gx_f, gy_f, deg)
                
                if has_collision(poly, existing_polys):
                    continue
                
                test_solution = existing + [(gx_f, gy_f, deg)]
                score = compute_bounding_square_side(test_solution)
                
                if score < best_score:
                    best_score = score
                    best_placement = (gx_f, gy_f, deg)
    
    # Finer grid search around best area
    if best_placement:
        bx, by, bd = best_placement
        for dx in [-0.3, -0.15, 0, 0.15, 0.3]:
            for dy in [-0.3, -0.15, 0, 0.15, 0.3]:
                for deg in rotations:
                    poly = make_tree_polygon(bx + dx, by + dy, deg)
                    if has_collision(poly, existing_polys):
                        continue
                    
                    test = existing + [(bx + dx, by + dy, deg)]
                    score = compute_bounding_square_side(test)
                    
                    if score < best_score:
                        best_score = score
                        best_placement = (bx + dx, by + dy, deg)
    
    if best_placement:
        return best_placement
    
    # Fallback: place outside
    return (max_x + 0.8, (min_y + max_y) / 2, 0.0)


def _repair_overlaps(placements: Solution, max_iters: int = 60, step: float = 0.06) -> Solution:
    """
    Resolve overlaps by gently repelling intersecting pairs.

    This is a deterministic, low-overhead relaxation that preserves the
    overall layout while guaranteeing an overlap-free configuration or
    falling back to a safe hexagonal pattern.
    """
    if not placements:
        return placements

    current = list(placements)
    for _ in range(max_iters):
        overlaps = check_all_overlaps(current)
        if not overlaps:
            return center_placements(current)

        adjustments = [(0.0, 0.0) for _ in current]
        for i, j in overlaps:
            xi, yi, ai = current[i]
            xj, yj, aj = current[j]
            dx, dy = xi - xj, yi - yj
            dist = math.hypot(dx, dy) or 1.0
            push = step / dist
            ax, ay = dx * push, dy * push
            adjustments[i] = (adjustments[i][0] + ax, adjustments[i][1] + ay)
            adjustments[j] = (adjustments[j][0] - ax, adjustments[j][1] - ay)

        current = [
            (x + ax, y + ay, ang)
            for (x, y, ang), (ax, ay) in zip(current, adjustments)
        ]

    # Fallback: guaranteed non-overlapping pattern
    return center_placements(initial_hexagonal_positions(len(placements)))


class PackingSolver:
    """
    Main solver using incremental building with SA optimization.
    
    Strategy:
    1. Start from N=1, incrementally add trees
    2. Use previous solution as warm start
    3. Optimize each N with SA
    4. Periodically try fresh starts to escape local minima
    """
    
    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        seed: int = 42,
        target_score: Optional[float] = None
    ):
        self.config = config or OptimizationConfig.standard_mode()
        self.seed = seed
        self.target_score = target_score
        self.target_per_tree: Optional[float] = None
        random.seed(seed)
        
        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}
    
    def solve_single(self, n: int, verbose: bool = False) -> Solution:
        """Solve for a single n value."""
        
        if n <= 0:
            return []
        
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = compute_bounding_square_side(sol)
            return sol
        
        # Ensure we have solution for n-1
        if n - 1 not in self.solutions:
            self.solve_single(n - 1, verbose=False)
        
        prev_solution = self.solutions[n - 1]
        prev_polys = [make_tree_polygon(x, y, d) for x, y, d in prev_solution]
        
        # Find best position for new tree
        new_placement = find_best_insertion_position(prev_solution, prev_polys)
        
        # Start with previous solution + new tree
        init_solution = prev_solution + [new_placement]
        
        # Validate no overlaps
        overlaps = check_all_overlaps(init_solution)
        if overlaps:
            # Fallback to grid if insertion caused overlap
            init_solution = initial_hexagonal_positions(n)
        
        # Scale iterations with n
        iterations_mult = 1.0 + (n / 80)
        
        # Optimize
        optimized = optimize_placement(
            init_solution,
            self.config,
            iterations_multiplier=iterations_mult,
            verbose=False
        )

        def maybe_improve(current: Solution) -> Solution:
            """Try alternative starts when we are above the target or stuck."""
            current_score = compute_bounding_square_side(current)
            per_tree_score = (current_score ** 2) / n
            needs_help = self.target_per_tree and per_tree_score > self.target_per_tree

            # Always allow a single alternative attempt for early trees to escape bad starts.
            if not needs_help and n > 30:
                return current

            alternatives = [
                initial_hexagonal_positions(n),
                initial_brick_positions(n),
                initial_spiral_positions(n),
            ]

            best_local = current
            best_score_local = current_score
            for alt in alternatives:
                alt_opt = optimize_placement(
                    alt,
                    self.config,
                    iterations_multiplier=iterations_mult * 0.6,
                    verbose=False
                )
                alt_score = compute_bounding_square_side(alt_opt)
                if alt_score < best_score_local:
                    best_score_local = alt_score
                    best_local = alt_opt
            return best_local

        optimized = maybe_improve(optimized)
        
        # Every 25 trees, try a fresh start to escape local minima
        if n % 25 == 0 and n > 25:
            alternatives = [
                initial_hexagonal_positions(n),
                initial_brick_positions(n),
                initial_spiral_positions(n),
            ]
            
            for alt in alternatives:
                alt_opt = optimize_placement(
                    alt,
                    self.config,
                    iterations_multiplier=iterations_mult * 0.5,
                    verbose=False
                )
                alt_score = compute_bounding_square_side(alt_opt)
                curr_score = compute_bounding_square_side(optimized)
                
                if alt_score < curr_score:
                    optimized = alt_opt
        
        # Final centering
        optimized = center_placements(optimized)
        optimized = _repair_overlaps(optimized)
        
        # Validate
        overlaps = check_all_overlaps(optimized)
        if overlaps:
            # Something went wrong, use safe fallback
            optimized = center_placements(initial_grid_positions(n))
        
        self.solutions[n] = optimized
        self.scores[n] = compute_bounding_square_side(optimized)
        
        if verbose and (n % 10 == 0 or n <= 10):
            print(f"  n={n}: side={self.scores[n]:.4f}")
        
        return optimized
    
    def solve_all(self, max_n: int = 200, verbose: bool = True) -> Dict[int, Solution]:
        """Solve for all n from 1 to max_n."""
        
        if verbose:
            print(f"Solving n=1 to {max_n}...")
            print(f"Config: {self.config.sa_iterations_base} base iterations, "
                  f"{self.config.num_restarts} restarts")

        self.target_per_tree = (
            self.target_score / max_n if self.target_score else None
        )

        def print_progress(n: int, side: float, total: float):
            width = 30
            filled = int(width * n / max_n)
            bar = "█" * filled + "░" * (width - filled)
            msg = (f"\r[{bar}] n={n}/{max_n} | side={side:.4f} "
                   f"| running score={total:.2f}")
            print(msg, end="", flush=True)
        
        running_total = 0.0
        for n in range(1, max_n + 1):
            self.solve_single(n, verbose=verbose)
            side = self.scores[n]
            running_total += (side ** 2) / n
            if verbose:
                print_progress(n, side, running_total)
        if verbose:
            print()
        
        total = self.compute_total_score()
        if verbose:
            print(f"\nTotal score: {total:.2f}")
        
        return self.solutions
    
    def compute_total_score(self) -> float:
        """Compute competition score: sum of (side^2 / n)."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, compute_bounding_square_side(sol))
            total += (side ** 2) / n
        return total
    
    def get_score_breakdown(self) -> Dict[int, float]:
        """Get individual scores per n."""
        breakdown = {}
        for n, sol in self.solutions.items():
            side = self.scores.get(n, compute_bounding_square_side(sol))
            breakdown[n] = (side ** 2) / n
        return breakdown


def create_initial_solution(n: int, strategy: str = "hexagonal") -> Solution:
    """Create initial solution with specified strategy."""
    if strategy == "grid":
        return initial_grid_positions(n)
    elif strategy == "brick":
        return initial_brick_positions(n)
    elif strategy == "spiral":
        return initial_spiral_positions(n)
    else:
        return initial_hexagonal_positions(n)
