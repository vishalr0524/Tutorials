import numpy as np

def dist(a, b):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# def dist_sq(a, b):
#     """Calculate squared Euclidean distance between two points."""
#     return (a[0]-b[0])**2 + (a[1]-b[1])**2

def compute_rule_based_intersection(pts):
    """
    MODIFIED LOGIC:
    1. top: Minimum Y coordinate.
    2. lowest: From remaining points, the one closest in X-coordinate to the top point's X-coordinate.
    3. next_top: The last remaining point.
    Returns top, next_top, lowest, ix, iy.
    """

    top = min(pts, key=lambda p: p[1])

    temp_pts = list(pts)
    temp_pts.remove(top)

    top_x = top[0]
    lowest = min(temp_pts, key=lambda p: abs(p[0] - top_x))
    
    temp_pts.remove(lowest)
    next_top = temp_pts[0]
    
    ix = top[0]
    
    x1, y1 = next_top
    x2, y2 = lowest
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0:
        iy = y1
    else:
        m = dy / dx
        iy = y1 + m * (ix - x1)
    
    return top, next_top, lowest, ix, iy

def get_triangle_details(top, next_top, lowest):
    """
    Identifies the hypotenuse, the third point, and the projection point.
    Returns:
        hyp_a, hyp_b: Endpoints of the hypotenuse
        third: The third point
        proj: The projection point on the hypotenuse
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
