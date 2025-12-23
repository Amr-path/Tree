"""
Santa 2025 Christmas Tree Packing Solver
Highly optimized version with:
- Correct 15-vertex tree polygon
- Aggressive Simulated Annealing
- Incremental building strategy
- Multiple initialization patterns
"""

from .geometry import (
    make_tree_polygon,
    transform_tree,
    compute_bounding_square_side,
    compute_bounding_box,
    check_all_overlaps,
    center_placements,
    TREE_WIDTH,
    TREE_HEIGHT,
)

from .optimize import (
    OptimizationConfig,
    optimize_layout,
    optimize_placement,
)

from .packing import (
    PackingSolver,
    initial_grid_positions,
    initial_hexagonal_positions,
    initial_brick_positions,
    create_initial_solution,
)

from .validate import (
    validate_solution,
    validate_all_solutions,
    compute_score,
    print_score_summary,
)

from .io_utils import (
    find_data_path,
    create_submission,
    print_solution_summary,
)

__version__ = "2.0.0"
