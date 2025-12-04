from flask import Flask, request, jsonify
import geometry
import numpy as np
import json
import time

app = Flask(__name__)

@app.route('/compute', methods=['POST'])
def compute():
    start_time = time.time()
    data = request.json
    print(data)
    pts = data.get('points')
    pixels_per_cm = data.get('pixels_per_cm')
    
    if not pts or len(pts) != 3:
        return jsonify({"error": "Invalid points"}), 400

    pts = [tuple(p) for p in pts]

    top, next_top, lowest, ix, iy = geometry.compute_rule_based_intersection(pts)
    hyp_a, hyp_b, third, proj = geometry.get_triangle_details(top, next_top, lowest)
    
    # Actual Throat
    len_actual_throat = geometry.dist(hyp_a, hyp_b) 
    
    # Leg1
    len_leg1 = geometry.dist((ix, iy), hyp_a)
    
    # Leg2
    len_leg2 = geometry.dist((ix, iy), hyp_b)
    
    # Root Penetration
    len_root_penetration = geometry.dist(third, (ix, iy)) 
    
    # Effective Throat
    len_effective_throat = geometry.dist(third, proj)
    
    # Format values
    def get_val_label(name, px_val):
        if pixels_per_cm:
            # pixels_per_mm = pixels_per_cm / 10
            cm_val = px_val / pixels_per_cm
            return {"value": cm_val, "unit": "cm", "label": f"{name}: {cm_val:.2f} cm"}
        else:
            return {"value": px_val, "unit": "px", "label": name}

    response = {
        "construction": {
            "top": top,
            "next_top": next_top,
            "lowest": lowest,
            "intersection": (ix, iy),
            "projection": proj,
            "hyp_a": hyp_a,
            "hyp_b": hyp_b,
            "third": third
        },
        "measurements": {
            "actual_throat": get_val_label("Actual Throat", len_actual_throat),
            "leg1": get_val_label("Leg1", len_leg1),
            "leg2": get_val_label("Leg2", len_leg2),
            "root_penetration": get_val_label("Root Penetration", len_root_penetration),
            "effective_throat": get_val_label("Effective Throat", len_effective_throat)
        }
    }
    print(response)
    end_time = time.time()    
    compute_time = end_time - start_time
    print(f"Time elapsed for a compute cycle: {compute_time * 1000:.3f} ms")
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
