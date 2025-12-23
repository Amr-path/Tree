"""
io_utils.py - File I/O utilities for Santa 2025
Handles submission format: id,x,y,deg with 's' prefix
"""
import os
from typing import List, Tuple, Dict, Optional

from .geometry import make_tree_polygon, compute_bounding_box

def find_data_path() -> str:
    """Find competition data directory."""
    candidates = [
        "/kaggle/input/santa-2025",
        "./data",
        "../data",
        "."
    ]
    for path in candidates:
        if os.path.isdir(path):
            sample = os.path.join(path, "sample_submission.csv")
            if os.path.isfile(sample):
                return path
    if os.path.isfile("sample_submission.csv"):
        return "."
    raise FileNotFoundError("Could not find competition data")

def get_output_path(filename: str = "submission.csv") -> str:
    """Get output path (Kaggle or local)."""
    if os.path.isdir("/kaggle/working"):
        return f"/kaggle/working/{filename}"
    return filename

def format_s_value(value: float, decimals: int = 6) -> str:
    """Format value with 's' prefix."""
    return f"s{value:.{decimals}f}"

def parse_s_value(s: str) -> float:
    """Parse 's' prefixed value."""
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def create_submission(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    output_path: Optional[str] = None,
    decimals: int = 6
) -> str:
    """
    Create submission CSV file.
    
    Format:
    - Header: id,x,y,deg
    - id: NNN_I (3-digit N, tree index I)
    - Values: sX.XXXXXX (6 decimals, 's' prefix)
    - Coordinates shifted so min x,y are at 0
    """
    if output_path is None:
        output_path = get_output_path()
    
    with open(output_path, "w") as f:
        f.write("id,x,y,deg\n")
        
        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing solution for n={n}")
            
            positions = solutions[n]
            if len(positions) != n:
                raise ValueError(f"Wrong count for n={n}: expected {n}, got {len(positions)}")
            
            # Get bounds and shift to origin
            polys = [make_tree_polygon(x, y, d) for x, y, d in positions]
            min_x = min(p.bounds[0] for p in polys)
            min_y = min(p.bounds[1] for p in polys)
            
            for idx, (x, y, deg) in enumerate(positions):
                # Shift coordinates
                X = x - min_x
                Y = y - min_y
                
                # Format with 's' prefix
                f.write(f"{n:03d}_{idx},"
                       f"s{X:.{decimals}f},"
                       f"s{Y:.{decimals}f},"
                       f"s{deg:.{decimals}f}\n")
    
    return output_path

def validate_submission_format(path: str) -> Tuple[bool, Optional[str]]:
    """Validate submission CSV format."""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return False, f"Cannot read file: {e}"
    
    if not lines:
        return False, "Empty file"
    
    # Check header
    header = lines[0].strip()
    if header != "id,x,y,deg":
        return False, f"Invalid header: {header}"
    
    # Check row count
    expected_rows = 200 * 201 // 2  # Sum 1..200
    actual_rows = len(lines) - 1
    if actual_rows != expected_rows:
        return False, f"Wrong row count: expected {expected_rows}, got {actual_rows}"
    
    # Spot check some rows
    for i in [1, 100, 1000, -1]:
        line = lines[i].strip()
        parts = line.split(',')
        if len(parts) != 4:
            return False, f"Invalid row format at line {i}"
        
        # Check 's' prefix
        for p in parts[1:]:
            if not p.startswith('s'):
                return False, f"Missing 's' prefix at line {i}"
    
    return True, None

def print_solution_summary(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200
):
    """Print solution summary."""
    from .geometry import compute_bounding_square_side
    from .validate import compute_score
    
    print("=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    
    n_solutions = len([n for n in range(1, max_n + 1) if n in solutions])
    print(f"Solutions: {n_solutions}/{max_n}")
    
    if n_solutions > 0:
        sides = {n: compute_bounding_square_side(solutions[n]) 
                 for n in range(1, max_n + 1) if n in solutions}
        
        print(f"\nBounding square range:")
        print(f"  Min: {min(sides.values()):.4f} (n={min(sides, key=sides.get)})")
        print(f"  Max: {max(sides.values()):.4f} (n={max(sides, key=sides.get)})")
        
        total = compute_score(solutions, max_n)
        print(f"\nTotal score: {total:.4f}")
        print(f"Baseline: ~157.08")
        if total < 157.08:
            print(f"Improvement: {(157.08 - total) / 157.08 * 100:.1f}%")
    
    print("=" * 60)
