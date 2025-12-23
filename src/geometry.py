"""
geometry.py - Core geometry for Santa 2025 with CORRECT 15-vertex tree polygon
From competition specs: trunk (0.15x0.20) + 3 canopy tiers (0.70, 0.40, 0.25 widths)
Total height: 1.0 (tip at y=0.8, trunk bottom at y=-0.2)
"""
import numpy as np
import math
from shapely.geometry import Polygon
from shapely import affinity
from shapely.prepared import prep
from shapely import STRtree
from typing import List, Tuple, Optional

# CORRECT 15-vertex tree polygon from competition (trunk top at origin)
TREE_COORDS = [
    (0.0, 0.8),        # tip
    (0.125, 0.5),      # top tier outer right
    (0.0625, 0.5),     # top tier inner right
    (0.2, 0.25),       # middle tier outer right
    (0.1, 0.25),       # middle tier inner right
    (0.35, 0.0),       # bottom tier outer right
    (0.075, 0.0),      # trunk top right
    (0.075, -0.2),     # trunk bottom right
    (-0.075, -0.2),    # trunk bottom left
    (-0.075, 0.0),     # trunk top left
    (-0.35, 0.0),      # bottom tier outer left
    (-0.1, 0.25),      # middle tier inner left
    (-0.2, 0.25),      # middle tier outer left
    (-0.0625, 0.5),    # top tier inner left
    (-0.125, 0.5),     # top tier outer left
]

TREE_VERTICES = np.array(TREE_COORDS)
BASE_TREE_POLYGON = Polygon(TREE_COORDS)

# Tree dimensions
TREE_WIDTH = 0.7   # max width at bottom tier
TREE_HEIGHT = 1.0  # tip to trunk bottom

def get_tree_polygon() -> Polygon:
    """Get the base tree polygon."""
    return BASE_TREE_POLYGON

def get_tree_area() -> float:
    return BASE_TREE_POLYGON.area

def make_tree_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """Create a Shapely Polygon for tree at (x,y) with rotation."""
    poly = BASE_TREE_POLYGON
    if angle_deg != 0:
        poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    if x != 0 or y != 0:
        poly = affinity.translate(poly, xoff=x, yoff=y)
    return poly

# Alias for compatibility
transform_tree = make_tree_polygon

def transform_vertices(x: float, y: float, deg: float) -> np.ndarray:
    """Fast numpy-based vertex transformation."""
    rad = math.radians(deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    return TREE_VERTICES @ rot.T + np.array([x, y])

def bounding_square_side(polygons: List[Polygon]) -> float:
    """Compute bounding square side length from polygon list."""
    if not polygons:
        return 0.0
    min_x = min(p.bounds[0] for p in polygons)
    min_y = min(p.bounds[1] for p in polygons)
    max_x = max(p.bounds[2] for p in polygons)
    max_y = max(p.bounds[3] for p in polygons)
    return max(max_x - min_x, max_y - min_y)

def compute_bounding_square_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side from placement list."""
    if not placements:
        return 0.0
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    return bounding_square_side(polys)

def compute_bounding_box(placements: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    if not placements:
        return (0, 0, 0, 0)
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    min_x = min(p.bounds[0] for p in polys)
    min_y = min(p.bounds[1] for p in polys)
    max_x = max(p.bounds[2] for p in polys)
    max_y = max(p.bounds[3] for p in polys)
    return min_x, min_y, max_x, max_y

def has_collision(tree_poly: Polygon, other_polys: List[Polygon]) -> bool:
    """Check if tree_poly overlaps any other polygon (touching OK)."""
    for poly in other_polys:
        if tree_poly.intersects(poly):
            if not tree_poly.touches(poly):
                return True
    return False

def check_overlap(p1: Polygon, p2: Polygon, eps: float = 1e-9) -> bool:
    """Check if two polygons overlap (touching OK)."""
    if not p1.intersects(p2):
        return False
    if p1.touches(p2):
        return False
    return p1.intersection(p2).area > eps

def check_all_overlaps(placements: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
    """Find all overlapping pairs."""
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if check_overlap(polys[i], polys[j]):
                overlaps.append((i, j))
    return overlaps

def center_placements(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Center placements around origin."""
    if not placements:
        return placements
    min_x, min_y, max_x, max_y = compute_bounding_box(placements)
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def normalize_to_origin(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Shift so min x,y are at 0."""
    if not placements:
        return placements
    min_x, min_y, _, _ = compute_bounding_box(placements)
    return [(x - min_x, y - min_y, d) for x, y, d in placements]

def placement_in_bounds(x: float, y: float, deg: float, limit: float = 100.0) -> bool:
    v = transform_vertices(x, y, deg)
    return np.all(v >= -limit) and np.all(v <= limit)

class CollisionDetector:
    """Fast collision detection using STRtree."""
    EPSILON = 1e-9
    
    def __init__(self):
        self.polygons: List[Polygon] = []
        self.tree: Optional[STRtree] = None
        self._dirty = True
    
    def clear(self):
        self.polygons = []
        self.tree = None
        self._dirty = True
    
    def add_polygon(self, poly: Polygon):
        self.polygons.append(poly)
        self._dirty = True
    
    def rebuild(self):
        if self._dirty and self.polygons:
            self.tree = STRtree(self.polygons)
            self._dirty = False
    
    def check_collision(self, poly: Polygon, exclude_index: int = -1) -> bool:
        self.rebuild()
        if not self.polygons:
            return False
        
        candidates = self.tree.query(poly)
        for idx in candidates:
            if idx == exclude_index:
                continue
            if poly.intersects(self.polygons[idx]):
                if not poly.touches(self.polygons[idx]):
                    return True
        return False
