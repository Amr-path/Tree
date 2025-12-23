"""
validate.py - Validation utilities for Santa 2025
"""
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .geometry import (
    make_tree_polygon, bounding_square_side, check_all_overlaps,
    compute_bounding_box, placement_in_bounds, compute_bounding_square_side
)

@dataclass
class ValidationResult:
    valid: bool
    n: int
    n_trees: int
    overlaps: List[Tuple[int, int]]
    out_of_bounds: List[int]
    bounding_square: float
    error_message: Optional[str] = None

def validate_solution(
    solution: List[Tuple[float, float, float]],
    n: int,
    bounds_limit: float = 100.0
) -> ValidationResult:
    """Validate a single solution."""
    
    if len(solution) != n:
        return ValidationResult(
            valid=False, n=n, n_trees=len(solution),
            overlaps=[], out_of_bounds=[],
            bounding_square=0.0,
            error_message=f"Expected {n} trees, got {len(solution)}"
        )
    
    if n == 0:
        return ValidationResult(
            valid=True, n=0, n_trees=0,
            overlaps=[], out_of_bounds=[],
            bounding_square=0.0
        )
    
    # Check bounds
    out_of_bounds = []
    for i, (x, y, d) in enumerate(solution):
        if not placement_in_bounds(x, y, d, bounds_limit):
            out_of_bounds.append(i)
    
    # Check overlaps
    overlaps = check_all_overlaps(solution)
    
    # Compute bounding square
    bounding_square = compute_bounding_square_side(solution)
    
    valid = len(overlaps) == 0 and len(out_of_bounds) == 0
    error_msg = None
    if not valid:
        errors = []
        if overlaps:
            errors.append(f"{len(overlaps)} overlap(s)")
        if out_of_bounds:
            errors.append(f"{len(out_of_bounds)} out of bounds")
        error_msg = "; ".join(errors)
    
    return ValidationResult(
        valid=valid, n=n, n_trees=len(solution),
        overlaps=overlaps, out_of_bounds=out_of_bounds,
        bounding_square=bounding_square,
        error_message=error_msg
    )

def validate_all_solutions(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200,
    verbose: bool = True
) -> bool:
    """Validate all solutions."""
    
    all_valid = True
    invalid_ns = []
    
    for n in range(1, max_n + 1):
        if n not in solutions:
            all_valid = False
            invalid_ns.append(n)
            continue
        
        result = validate_solution(solutions[n], n)
        if not result.valid:
            all_valid = False
            invalid_ns.append(n)
    
    if verbose:
        if all_valid:
            print(f"✓ All {max_n} solutions are valid")
        else:
            print(f"✗ {len(invalid_ns)} invalid solution(s)")
            for n in invalid_ns[:10]:
                if n in solutions:
                    result = validate_solution(solutions[n], n)
                    print(f"  n={n}: {result.error_message}")
                else:
                    print(f"  n={n}: Missing")
    
    return all_valid

def compute_score(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200
) -> float:
    """Compute total competition score."""
    total = 0.0
    for n in range(1, max_n + 1):
        if n not in solutions:
            continue
        side = compute_bounding_square_side(solutions[n])
        total += (side ** 2) / n
    return total

def print_score_summary(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200
):
    """Print detailed score summary."""
    
    print("=" * 60)
    print("SCORE SUMMARY")
    print("=" * 60)
    
    sides = {}
    contributions = {}
    
    for n in range(1, max_n + 1):
        if n not in solutions:
            continue
        side = compute_bounding_square_side(solutions[n])
        sides[n] = side
        contributions[n] = (side ** 2) / n
    
    total = sum(contributions.values())
    
    print(f"Total score: {total:.4f}")
    print(f"Baseline score: ~157.08")
    print(f"Improvement: {(157.08 - total) / 157.08 * 100:.1f}%")
    print()
    
    # Show worst contributors
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 worst contributions:")
    for n, contrib in sorted_contrib[:10]:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contrib={contrib:.4f}")
    
    print()
    print(f"Average side length: {sum(sides.values()) / len(sides):.4f}")
    print(f"Min side (n=1): {sides.get(1, 0):.4f}")
    print(f"Max side (n={max_n}): {sides.get(max_n, 0):.4f}")
    print("=" * 60)
