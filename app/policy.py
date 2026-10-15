"""Explicit hunt/bisection IDA; generic numerical failures are never crossings."""
import math
from common import THRESHOLDS
from fragility_poc.fishbone_ida import estimate_limit_state_capacities

def decide(points, policy):
    caps = estimate_limit_state_capacities(points, THRESHOLDS,
                          bracket_tolerance=policy['bracket_tolerance'])
    if points and points[-1]['status'] == 'numerical_failure':
        return {'state': 'needs_review', 'reason': 'numerical_failure_not_a_crossing', 'capacities': caps}
    if all(c['certified'] for c in caps.values()):
        return {'state': 'complete', 'capacities': caps}
    if len(points) >= policy['maximum_points_per_curve']:
        return {'state': 'needs_review', 'reason': 'point_guard_no_automatic_extension', 'capacities': caps}
    if not points:
        im = policy['initial_im_g']
    else:
        valid = [p for p in points if p['status'] in ('success','cp_endpoint_reached')]
        unfinished = [c for c in caps.values() if not c['certified']]
        bracketed = [c for c in unfinished if c['status'] == 'bracketed']
        if bracketed:
            c = max(bracketed, key=lambda c: c['relative_bracket_width'])
            im = math.sqrt(c['lower_im_g'] * c['upper_im_g'])
        else:
            # If the first evaluated point already crosses IO, find a genuine lower.
            if not any(p['status'] == 'success' and p['max_midr'] < .01 for p in valid):
                im = min(p['target_im_g'] for p in valid) / policy['hunt_multiplier']
            else:
                im = max(p['target_im_g'] for p in valid) * policy['hunt_multiplier']
    if not policy['minimum_im_g'] <= im <= policy['maximum_im_g']:
        return {'state': 'needs_review', 'reason': 'IM_guard_unbracketed_not_capacity', 'capacities': caps}
    if any(abs(math.log(im / p['target_im_g'])) < 1e-10 for p in points):
        return {'state': 'needs_review', 'reason': 'duplicate_IM_refused', 'capacities': caps}
    return {'state': 'continue', 'next_im_g': im, 'capacities': caps}
