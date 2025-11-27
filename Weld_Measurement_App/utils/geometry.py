import math
from PyQt6.QtCore import QPointF, QLineF

def get_line_intersection(line1, line2):
    """
    Returns the intersection point of two QLineF objects.
    Returns None if lines are parallel.
    """
    type, intersection_point = line1.intersects(line2)
    if type == QLineF.IntersectionType.BoundedIntersection:
        return intersection_point
    # We might want UnboundedIntersection too if lines don't physically cross but their extensions do?
    # User said "if two lines are crossed", usually implies visual crossing.
    # But for weld calcs, legs might extend. Let's allow Unbounded for calculation purposes if needed, 
    # but for "marking" intersections, visual crossing (Bounded) is usually what's meant.
    # Let's stick to Bounded for now as per "crossed".
    return None

def distance(p1, p2):
    """Euclidean distance between two QPointF."""
    dx = p1.x() - p2.x()
    dy = p1.y() - p2.y()
    return math.sqrt(dx*dx + dy*dy)

def midpoint(p1, p2):
    """Returns the midpoint QPointF between p1 and p2."""
    return QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
