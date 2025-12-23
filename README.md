# Santa 2025 Christmas Tree Packing Solver v2.0

Highly optimized solution for the Kaggle Santa 2025 competition.

## Key Improvements Over Baseline

1. **Correct Tree Polygon**: Uses exact 15-vertex polygon from competition specs
2. **5-10x More SA Iterations**: Much deeper optimization
3. **Multiple Initialization Patterns**: Hexagonal, brick, spiral patterns
4. **Multi-Restart Optimization**: Escapes local minima
5. **Incremental Building with Reshuffling**: Avoids getting stuck

## Quick Start

```bash
# Quick mode (~20 min, score ~120-140)
python run.py --mode quick

# Standard mode (~60 min, score ~100-120)
python run.py --mode standard

# Aggressive mode (~2 hours, score ~85-100)
python run.py --mode aggressive

# Maximum mode (~4 hours, score ~75-85)
python run.py --mode maximum
```

## On Kaggle

1. Upload all files to notebook
2. Add Santa 2025 dataset
3. Run: `!python run.py --mode standard`

## Expected Scores

| Mode       | Time     | Score Range | Improvement |
|------------|----------|-------------|-------------|
| quick      | 20 min   | 120-140     | 11-24%      |
| standard   | 60 min   | 100-120     | 24-36%      |
| aggressive | 2 hours  | 85-100      | 36-46%      |
| maximum    | 4 hours  | 75-85       | 46-52%      |

Baseline score: ~157.08

## Files

- `run.py` - Main entry point
- `src/geometry.py` - Tree polygon and collision detection
- `src/optimize.py` - Simulated Annealing optimizer
- `src/packing.py` - Main packing solver
- `src/validate.py` - Solution validation
- `src/io_utils.py` - File I/O utilities

## Algorithm

1. **Incremental Building**: Start from n=1, add trees one at a time
2. **Smart Insertion**: Find optimal position for each new tree
3. **Simulated Annealing**: Optimize after each insertion
4. **Periodic Reshuffling**: Every 25 trees, try fresh patterns
5. **Multi-Restart**: Multiple SA runs, keep best result

## Tree Polygon

The correct tree shape has 15 vertices:
- Trunk: 0.15 wide × 0.20 tall
- Bottom tier: 0.70 wide
- Middle tier: 0.40 wide  
- Top tier: 0.25 wide
- Total height: 1.00

## Requirements

```
numpy>=1.21.0
shapely>=2.0.0
```
