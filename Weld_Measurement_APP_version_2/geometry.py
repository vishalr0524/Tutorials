import numpy as np

def dist(a, b):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def compute_projection(A, B, P):
    """
    Calculates the projection point of P onto the infinite line defined by A and B.
    A and B are points on the line, P is the point to project.
    Returns (proj_x, proj_y)
    """
    x1, y1 = A
    x2, y2 = B
    x0, y0 = P

    AB = np.array([x2 - x1, y2 - y1], dtype=float)
    AP = np.array([x0 - x1, y0 - y1], dtype=float)

    denom = np.dot(AB, AB)
    if denom == 0:
        # A and B are the same point; projection is P
        return (x0, y0)
    
    # t is the scalar projection along vector AB
    t = np.dot(AP, AB) / denom
        
    proj_x = x1 + t * AB[0]
    proj_y = y1 + t * AB[1]
    
    # Keep as float for precision before converting to int when necessary
    return (proj_x, proj_y)

def compute_rule_based_intersection(pts):
    """
    MODIFIED LOGIC:
    Calculates intersection (ix, iy) as the PERPENDICULAR PROJECTION
    of the top point onto the line defined by next_top and lowest.
    """

    top = min(pts, key=lambda p: p[1])

    temp_pts = list(pts)
    temp_pts.remove(top)

    top_x = top[0]
    lowest = min(temp_pts, key=lambda p: abs(p[0] - top_x))
    
    temp_pts.remove(lowest)
    next_top = temp_pts[0]
    
    # --- MODIFICATION: Calculate intersection as PERPENDICULAR PROJECTION ---
    # Projection of Top point onto the line (Next Top - Lowest)
    ix_float, iy_float = compute_projection(next_top, lowest, top) 
    ix, iy = int(round(ix_float)), int(round(iy_float))
    
    return top, next_top, lowest, ix, iy

def get_triangle_details(top, next_top, lowest):
    """
    Identifies the hypotenuse, the third point, and the projection point.
    (No changes here)
    """
    hyp_a = top
    hyp_b = next_top
    third = lowest

    x1, y1 = hyp_a
    x2, y2 = hyp_b
    x0, y0 = third

    AB = np.array([x2 - x1, y2 - y1], dtype=float)
    AP = np.array([x0 - x1, y0 - y1], dtype=float)

    denom = np.dot(AB, AB)
    if denom == 0:
        t = 0
    else:
        t = np.dot(AP, AB) / denom
        
    t = max(0, min(1, t)) 

    proj_x = x1 + t * AB[0]
    proj_y = y1 + t * AB[1]
    
    proj = (int(proj_x), int(proj_y))
    
    return hyp_a, hyp_b, third, proj