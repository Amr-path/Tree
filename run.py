#!/usr/bin/env python3
"""
run.py - Main entry point for Santa 2025 Solver

Usage:
    python run.py [--mode quick|standard|aggressive|maximum] [--seed 42]

Modes:
    quick:      ~15-30 min, score ~120-140
    standard:   ~45-60 min, score ~100-120  
    aggressive: ~90-120 min, score ~85-100
    maximum:    ~3-4 hours, score ~75-85
"""
import sys
import os
import time
import argparse

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.packing import PackingSolver
from src.optimize import OptimizationConfig
from src.validate import validate_all_solutions, compute_score, print_score_summary
from src.io_utils import (
    find_data_path, create_submission, get_output_path,
    validate_submission_format, print_solution_summary
)


def get_config(mode: str) -> OptimizationConfig:
    """Get configuration for specified mode."""
    if mode == "quick":
        return OptimizationConfig.quick_mode()
    elif mode == "standard":
        return OptimizationConfig.standard_mode()
    elif mode == "aggressive":
        return OptimizationConfig.aggressive_mode()
    elif mode == "maximum":
        return OptimizationConfig.maximum_mode()
    else:
        return OptimizationConfig.standard_mode()


def main(mode: str = "standard", seed: int = 42, max_n: int = 200):
    """Main solver entry point."""
    
    print("=" * 70)
    print("SANTA 2025 CHRISTMAS TREE PACKING SOLVER v2.0")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print(f"Max N: {max_n}")
    print()
    
    # Find data
    try:
        data_path = find_data_path()
        print(f"✓ Data path: {data_path}")
    except FileNotFoundError:
        print("Note: Competition data not found (OK for solving)")
    
    # Get config
    config = get_config(mode)
    config.seed = seed
    
    print(f"\nConfiguration:")
    print(f"  Base iterations: {config.sa_iterations_base}")
    print(f"  Num restarts: {config.num_restarts}")
    print(f"  Temp factor: {config.temp_initial_factor}")
    print()
    
    # Solve
    start_time = time.time()
    
    print(f"Solving n=1 to {max_n}...")
    print("-" * 50)
    
    solver = PackingSolver(config=config, seed=seed)
    solutions = solver.solve_all(max_n=max_n, verbose=True)
    
    solve_time = time.time() - start_time
    print("-" * 50)
    print(f"Solving completed in {solve_time:.1f}s ({solve_time/60:.1f} min)")
    
    # Validate
    print("\nValidating solutions...")
    all_valid = validate_all_solutions(solutions, max_n=max_n, verbose=True)
    
    # Score summary
    print()
    print_score_summary(solutions, max_n=max_n)
    
    # Create submission
    print("\nCreating submission...")
    output_path = get_output_path("submission.csv")
    
    try:
        created_path = create_submission(solutions, output_path=output_path)
        print(f"✓ Saved to: {created_path}")
        
        is_valid, error = validate_submission_format(created_path)
        if is_valid:
            print("✓ Format validated")
        else:
            print(f"⚠ Format issue: {error}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    total_score = compute_score(solutions, max_n)
    
    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total score: {total_score:.4f}")
    print(f"Baseline: 157.08")
    print(f"Improvement: {(157.08 - total_score) / 157.08 * 100:.1f}%")
    print(f"Output: {output_path}")
    print()
    
    return solutions


def parse_args():
    parser = argparse.ArgumentParser(description="Santa 2025 Solver")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "aggressive", "maximum"],
        default="standard",
        help="Optimization mode"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-n", type=int, default=200, help="Max N to solve")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(mode=args.mode, seed=args.seed, max_n=args.max_n)
