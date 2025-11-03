""" Python implementation of
    Algorithm for Automatically Fitting Digitized Curves
    by Philip J. Schneider
    "Graphics Gems", Academic Press, 1990
    Copy from https://github.com/zhengzheng/fitCurves.git
"""
from __future__ import print_function
from numpy import linalg, dot, zeros
import numpy as np


# import bezier
"""
Evaluates a cubic Bezier curve at parameter t, returning the point coordinates.

Parameters:
    ctrl_poly: Array of 4 control points defining the Bezier curve
    t: Parameter value in the range [0,1]

Returns:
    The point coordinates on the Bezier curve corresponding to parameter t
"""
def q(ctrl_poly, t):
    return (1.0-t)**3 * ctrl_poly[0] + 3*(1.0-t)**2 * t * ctrl_poly[1] + 3*(1.0-t)* t**2 * ctrl_poly[2] + t**3 * ctrl_poly[3]


"""
Evaluates the first derivative of a cubic Bezier curve at parameter t.

Parameters:
    ctrl_poly: Array of 4 control points defining the Bezier curve
    t: Parameter value in the range [0,1]

Returns:
    The first derivative (tangent vector) of the Bezier curve at parameter t
"""
def qprime(ctrl_poly, t):
    return 3*(1.0-t)**2 * (ctrl_poly[1]-ctrl_poly[0]) + 6*(1.0-t) * t * (ctrl_poly[2]-ctrl_poly[1]) + 3*t**2 * (ctrl_poly[3]-ctrl_poly[2])


"""
Evaluates the second derivative of a cubic Bezier curve at parameter t.

Parameters:
    ctrl_poly: Array of 4 control points defining the Bezier curve
    t: Parameter value in the range [0,1]

Returns:
    The second derivative of the Bezier curve at parameter t
"""
def qprimeprime(ctrl_poly, t):
    return 6*(1.0-t) * (ctrl_poly[2]-2*ctrl_poly[1]+ctrl_poly[0]) + 6*(t) * (ctrl_poly[3]-2*ctrl_poly[2]+ctrl_poly[1])

"""
Main function to fit one or more Bezier curves to a set of points.

Parameters:
    points: Array of points to be fitted
    max_error: Maximum allowable fitting error

Returns:
    Array of fitted Bezier curves, each represented by 4 control points
"""
def fit_curve(points, max_error):
    if (len(points) <= 1):
        return []
    left_tangent = normalize(points[1] - points[0])
    right_tangent = normalize(points[-2] - points[-1])
    return fit_cubic(points, left_tangent, right_tangent, max_error)

"""
def fit_curve_with_max_segments(points, max_error, max_segments):
    if (len(points) <= 1):
        return []
    left_tangent = normalize(points[1] - points[0])
    right_tangent = normalize(points[-2] - points[-1])
    beziers = fit_cubic(points, left_tangent, right_tangent, max_error)
    while len(beziers) > max_segments:
        max_error *= 1.5
        beziers = fit_cubic(points, left_tangent, right_tangent, max_error)
    return beziers

def fit_one_cubic(all_points, start_index, end_index, left_tangent, right_tangent, error):
    points = all_points[start_index:end_index+1]
    if (len(points) == 2):
        dist = linalg.norm(points[0] - points[1]) / 3.0
        bez_curve = [points[0], points[0] + left_tangent * dist, points[1] + right_tangent * dist, points[1]]
        return bez_curve, 0.0, 0
    # Parameterize points, and attempt to fit curve
    u = chord_length_parameterize(points)
    bez_curve = generate_bezier(points, u, left_tangent, right_tangent)
    # Find max deviation of points to fitted curve
    max_error, split_point = compute_max_error(points, bez_curve, u)
    if max_error < error:
        return bez_curve, max_error, split_point + start_index
    
    for _ in range(20):
        u_prime = reparameterize(bez_curve, points, u)
        bez_curve = generate_bezier(points, u_prime, left_tangent, right_tangent)
        max_error, split_point = compute_max_error(points, bez_curve, u_prime)
        if max_error < error:
            return bez_curve, max_error, split_point + start_index
        u = u_prime

    return bez_curve, max_error, split_point + start_index


def fit_cubic_with_max_segments(points, left_tangent, right_tangent, error, max_splits):
    # Use heuristic if region only has two points in it
    max_segments = 2 ** max_splits
    if (len(points) == 2):
        dist_for_each_segment = linalg.norm(points[0] - points[1]) / max_segments
        dist = dist_for_each_segment / 3.0
        bez_curves = []
        for i in range(max_segments):
            start_point = points[0] + left_tangent * dist_for_each_segment * i
            ctrl1 = start_point + left_tangent * dist
            end_point = points[0] + left_tangent * dist_for_each_segment * (i+1)
            ctrl2 = end_point + right_tangent * dist
            bez_curve = [start_point, ctrl1, ctrl2, end_point]
            bez_curves.append(bez_curve)
        return bez_curves

    beziers = []
    errors = []
    split_points = []
    while len(beziers) < max_segments:
        if len(beziers) == 0:
            bez_curve, max_error, split_point = fit_one_cubic(points, 0, len(points)-1, left_tangent, right_tangent, error)
            beziers.append(bez_curve)
            errors.append(max_error)
            split_points.append([0, split_point, len(points)-1])
        else:
            if max(errors) == 0:
                rnd_index = np.random.randint(0, len(beziers))
                max_error_index = rnd_index
            else:
                max_error_index = errors.index(max(errors))
            start_index, split_point, end_index = split_points[max_error_index]
            center_tangent = normalize(points[split_point-1] - points[split_point+1])
            bez_curve1, max_error1, split_point1 = fit_one_cubic(points, start_index, split_point, left_tangent, center_tangent, error)
            bez_curve2, max_error2, split_point2 = fit_one_cubic(points, split_point, end_index, -center_tangent, right_tangent, error)
            beziers[max_error_index] = bez_curve1
            errors[max_error_index] = max_error1
            split_points[max_error_index] = [start_index, split_point1, split_point]
            beziers.append(bez_curve2)
            errors.append(max_error2)
            split_points.append([split_point, split_point2, end_index])
            
    return beziers
"""



"""
Fits a set of points to a cubic Bezier curve using specified left and right tangent directions.

Parameters:
    points: Array of points to be fitted
    left_tangent: Tangent direction at the start point
    right_tangent: Tangent direction at the end point
    error: Allowable fitting error

Returns:
    Array of fitted Bezier curves
"""
def fit_cubic(points, left_tangent, right_tangent, error):
    # Use heuristic if region only has two points in it
    if (len(points) == 2):
        dist = linalg.norm(points[0] - points[1]) / 3.0
        bez_curve = [points[0], points[0] + left_tangent * dist, points[1] + right_tangent * dist, points[1]]
        return [bez_curve]

    # Parameterize points, and attempt to fit curve
    u = chord_length_parameterize(points)
    bez_curve = generate_bezier(points, u, left_tangent, right_tangent)
    # Find max deviation of points to fitted curve
    max_error, split_point = compute_max_error(points, bez_curve, u)
    if max_error < error:
        return [bez_curve]

    # If error not too large, try some reparameterization and iteration
    if max_error < error**2:
        for _ in range(20):
            u_prime = reparameterize(bez_curve, points, u)
            bez_curve = generate_bezier(points, u_prime, left_tangent, right_tangent)
            max_error, split_point = compute_max_error(points, bez_curve, u_prime)
            if max_error < error:
                return [bez_curve]
            u = u_prime

    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    center_tangent = normalize(points[split_point-1] - points[split_point+1])
    beziers += fit_cubic(points[:split_point+1], left_tangent, center_tangent, error)
    beziers += fit_cubic(points[split_point:], -center_tangent, right_tangent, error)

    return beziers


"""
Generates a Bezier curve based on given points, parameters, and tangent directions.

Parameters:
    points: Array of points
    parameters: Array of parameterized values
    left_tangent: Left tangent direction
    right_tangent: Right tangent direction

Returns:
    Array of control points defining the generated Bezier curve
"""
def generate_bezier(points, parameters, left_tangent, right_tangent):
    bez_curve = [points[0], None, None, points[-1]]

    # compute the A's
    A = zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = left_tangent  * 3*(1-u)**2 * u
        A[i][1] = right_tangent * 3*(1-u)    * u**2

    # Create the C and X matrices
    C = zeros((2, 2))
    X = zeros(2)

    for i, (point, u) in enumerate(zip(points, parameters)):
        C[0][0] += dot(A[i][0], A[i][0])
        C[0][1] += dot(A[i][0], A[i][1])
        C[1][0] += dot(A[i][0], A[i][1])
        C[1][1] += dot(A[i][1], A[i][1])

        tmp = point - q([points[0], points[0], points[-1], points[-1]], u)

        X[0] += dot(A[i][0], tmp)
        X[1] += dot(A[i][1], tmp)

    # Compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X  = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1  = X[0] * C[1][1] - X[1] * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text) */
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent newton_raphson_root_find() call. */
    seg_length = linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * seg_length
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bez_curve[1] = bez_curve[0] + left_tangent * (seg_length / 3.0)
        bez_curve[2] = bez_curve[3] + right_tangent * (seg_length / 3.0)

    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bez_curve[1] = bez_curve[0] + left_tangent * alpha_l
        bez_curve[2] = bez_curve[3] + right_tangent * alpha_r

    return bez_curve


"""
Reparameterizes points on a Bezier curve using Newton-Raphson method.

Parameters:
    bezier: Array of control points defining the Bezier curve
    points: Array of points
    parameters: Current parameter array

Returns:
    Optimized parameter array
"""
def reparameterize(bezier, points, parameters):
    return [newton_raphson_root_find(bezier, point, u) for point, u in zip(points, parameters)]


"""
Uses Newton-Raphson method to find the parameter value on the curve that minimizes
the distance to a given point.

Newton's root finding algorithm calculates f(x)=0 by reiterating x_n+1 = x_n - f(x_n)/f'(x_n)

We are trying to find curve parameter u for some point p that minimizes
the distance from that point to the curve. At minimum distance the point is perpendicular to the curve.
We are solving f = q(u)-p * q'(u) = 0
with f' = q'(u) * q'(u) + q(u)-p * q''(u)

which gives u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|

Parameters:
    bez: Array of control points defining the Bezier curve
    point: Given point
    u: Initial parameter value

Returns:
    Optimized parameter value
"""
def newton_raphson_root_find(bez, point, u):
    d = q(bez, u)-point
    numerator = (d * qprime(bez, u)).sum()
    denominator = (qprime(bez, u)**2 + d * qprimeprime(bez, u)).sum()

    if denominator == 0.0:
        return u
    else:
        return u - numerator/denominator


"""
Parameterizes a set of points based on chord length.

Parameters:
    points: Array of points

Returns:
    Parameterized array with values in range [0,1]
"""
def chord_length_parameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i-1] + linalg.norm(points[i] - points[i-1]))

    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]

    return u


"""
Computes the maximum error and corresponding split point between a set of points and a Bezier curve.

Parameters:
    points: Array of points
    bez: Array of control points defining the Bezier curve
    parameters: Parameter array

Returns:
    Tuple containing (maximum error value, index of the point with maximum error)
"""
def compute_max_error(points, bez, parameters):
    max_dist = 0.0
    split_point = len(points)/2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = linalg.norm(q(bez, u)-point)**2
        if dist > max_dist:
            max_dist = dist
            split_point = i

    return max_dist, split_point


"""
Normalizes a vector to unit length.

Parameters:
    v: Input vector

Returns:
    Normalized unit vector
"""
def normalize(v):
    return v / linalg.norm(v)
