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
    ctrlPoly: Array of 4 control points defining the Bezier curve
    t: Parameter value in the range [0,1]

Returns:
    The point coordinates on the Bezier curve corresponding to parameter t
"""
def q(ctrlPoly, t):
    return (1.0-t)**3 * ctrlPoly[0] + 3*(1.0-t)**2 * t * ctrlPoly[1] + 3*(1.0-t)* t**2 * ctrlPoly[2] + t**3 * ctrlPoly[3]


"""
Evaluates the first derivative of a cubic Bezier curve at parameter t.

Parameters:
    ctrlPoly: Array of 4 control points defining the Bezier curve
    t: Parameter value in the range [0,1]

Returns:
    The first derivative (tangent vector) of the Bezier curve at parameter t
"""
def qprime(ctrlPoly, t):
    return 3*(1.0-t)**2 * (ctrlPoly[1]-ctrlPoly[0]) + 6*(1.0-t) * t * (ctrlPoly[2]-ctrlPoly[1]) + 3*t**2 * (ctrlPoly[3]-ctrlPoly[2])


"""
Evaluates the second derivative of a cubic Bezier curve at parameter t.

Parameters:
    ctrlPoly: Array of 4 control points defining the Bezier curve
    t: Parameter value in the range [0,1]

Returns:
    The second derivative of the Bezier curve at parameter t
"""
def qprimeprime(ctrlPoly, t):
    return 6*(1.0-t) * (ctrlPoly[2]-2*ctrlPoly[1]+ctrlPoly[0]) + 6*(t) * (ctrlPoly[3]-2*ctrlPoly[2]+ctrlPoly[1])

"""
Main function to fit one or more Bezier curves to a set of points.

Parameters:
    points: Array of points to be fitted
    maxError: Maximum allowable fitting error

Returns:
    Array of fitted Bezier curves, each represented by 4 control points
"""
def fitCurve(points, maxError):
    if (len(points) <= 1):
        return []
    leftTangent = normalize(points[1] - points[0])
    rightTangent = normalize(points[-2] - points[-1])
    return fitCubic(points, leftTangent, rightTangent, maxError)

"""
def fitCurveWithMaxSegments(points, maxError, maxSegments):
    if (len(points) <= 1):
        return []
    leftTangent = normalize(points[1] - points[0])
    rightTangent = normalize(points[-2] - points[-1])
    beziers = fitCubic(points, leftTangent, rightTangent, maxError)
    while len(beziers) > maxSegments:
        maxError *= 1.5
        beziers = fitCubic(points, leftTangent, rightTangent, maxError)
    return beziers

def fitOneCubic(all_points, start_index, end_index, leftTangent, rightTangent, error):
    points = all_points[start_index:end_index+1]
    if (len(points) == 2):
        dist = linalg.norm(points[0] - points[1]) / 3.0
        bezCurve = [points[0], points[0] + leftTangent * dist, points[1] + rightTangent * dist, points[1]]
        return bezCurve, 0.0, 0
    # Parameterize points, and attempt to fit curve
    u = chordLengthParameterize(points)
    bezCurve = generateBezier(points, u, leftTangent, rightTangent)
    # Find max deviation of points to fitted curve
    maxError, splitPoint = computeMaxError(points, bezCurve, u)
    if maxError < error:
        return bezCurve, maxError, splitPoint + start_index
    
    for _ in range(20):
        uPrime = reparameterize(bezCurve, points, u)
        bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent)
        maxError, splitPoint = computeMaxError(points, bezCurve, uPrime)
        if maxError < error:
            return bezCurve, maxError, splitPoint + start_index
        u = uPrime

    return bezCurve, maxError, splitPoint + start_index


def fitCubicWithMaxSegments(points, leftTangent, rightTangent, error, maxSplits):
    # Use heuristic if region only has two points in it
    maxSegments = 2 ** maxSplits
    if (len(points) == 2):
        dist_for_each_segment = linalg.norm(points[0] - points[1]) / maxSegments
        dist = dist_for_each_segment / 3.0
        bezCurves = []
        for i in range(maxSegments):
            start_point = points[0] + leftTangent * dist_for_each_segment * i
            ctrl1 = start_point + leftTangent * dist
            end_point = points[0] + leftTangent * dist_for_each_segment * (i+1)
            ctrl2 = end_point + rightTangent * dist
            bezCurve = [start_point, ctrl1, ctrl2, end_point]
            bezCurves.append(bezCurve)
        return bezCurves

    beziers = []
    errors = []
    splitPoints = []
    while len(beziers) < maxSegments:
        if len(beziers) == 0:
            bezCurve, maxError, splitPoint = fitOneCubic(points, 0, len(points)-1, leftTangent, rightTangent, error)
            beziers.append(bezCurve)
            errors.append(maxError)
            splitPoints.append([0, splitPoint, len(points)-1])
        else:
            if max(errors) == 0:
                rnd_index = np.random.randint(0, len(beziers))
                maxErrorIndex = rnd_index
            else:
                maxErrorIndex = errors.index(max(errors))
            start_index, splitPoint, end_index = splitPoints[maxErrorIndex]
            centerTengent = normalize(points[splitPoint-1] - points[splitPoint+1])
            bezCurve1, maxError1, splitPoint1 = fitOneCubic(points, start_index, splitPoint, leftTangent, centerTengent, error)
            bezCurve2, maxError2, splitPoint2 = fitOneCubic(points, splitPoint, end_index, -centerTengent, rightTangent, error)
            beziers[maxErrorIndex] = bezCurve1
            errors[maxErrorIndex] = maxError1
            splitPoints[maxErrorIndex] = [start_index, splitPoint1, splitPoint]
            beziers.append(bezCurve2)
            errors.append(maxError2)
            splitPoints.append([splitPoint, splitPoint2, end_index])
            
    return beziers
"""



"""
Fits a set of points to a cubic Bezier curve using specified left and right tangent directions.

Parameters:
    points: Array of points to be fitted
    leftTangent: Tangent direction at the start point
    rightTangent: Tangent direction at the end point
    error: Allowable fitting error

Returns:
    Array of fitted Bezier curves
"""
def fitCubic(points, leftTangent, rightTangent, error):
    # Use heuristic if region only has two points in it
    if (len(points) == 2):
        dist = linalg.norm(points[0] - points[1]) / 3.0
        bezCurve = [points[0], points[0] + leftTangent * dist, points[1] + rightTangent * dist, points[1]]
        return [bezCurve]

    # Parameterize points, and attempt to fit curve
    u = chordLengthParameterize(points)
    bezCurve = generateBezier(points, u, leftTangent, rightTangent)
    # Find max deviation of points to fitted curve
    maxError, splitPoint = computeMaxError(points, bezCurve, u)
    if maxError < error:
        return [bezCurve]

    # If error not too large, try some reparameterization and iteration
    if maxError < error**2:
        for _ in range(20):
            uPrime = reparameterize(bezCurve, points, u)
            bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent)
            maxError, splitPoint = computeMaxError(points, bezCurve, uPrime)
            if maxError < error:
                return [bezCurve]
            u = uPrime

    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    centerTangent = normalize(points[splitPoint-1] - points[splitPoint+1])
    beziers += fitCubic(points[:splitPoint+1], leftTangent, centerTangent, error)
    beziers += fitCubic(points[splitPoint:], -centerTangent, rightTangent, error)

    return beziers


"""
Generates a Bezier curve based on given points, parameters, and tangent directions.

Parameters:
    points: Array of points
    parameters: Array of parameterized values
    leftTangent: Left tangent direction
    rightTangent: Right tangent direction

Returns:
    Array of control points defining the generated Bezier curve
"""
def generateBezier(points, parameters, leftTangent, rightTangent):
    bezCurve = [points[0], None, None, points[-1]]

    # compute the A's
    A = zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = leftTangent  * 3*(1-u)**2 * u
        A[i][1] = rightTangent * 3*(1-u)    * u**2

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
    # divide by zero in any subsequent NewtonRaphsonRootFind() call. */
    segLength = linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * segLength
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bezCurve[1] = bezCurve[0] + leftTangent * (segLength / 3.0)
        bezCurve[2] = bezCurve[3] + rightTangent * (segLength / 3.0)

    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bezCurve[1] = bezCurve[0] + leftTangent * alpha_l
        bezCurve[2] = bezCurve[3] + rightTangent * alpha_r

    return bezCurve


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
    return [newtonRaphsonRootFind(bezier, point, u) for point, u in zip(points, parameters)]


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
def newtonRaphsonRootFind(bez, point, u):
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
def chordLengthParameterize(points):
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
def computeMaxError(points, bez, parameters):
    maxDist = 0.0
    splitPoint = len(points)/2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = linalg.norm(q(bez, u)-point)**2
        if dist > maxDist:
            maxDist = dist
            splitPoint = i

    return maxDist, splitPoint


"""
Normalizes a vector to unit length.

Parameters:
    v: Input vector

Returns:
    Normalized unit vector
"""
def normalize(v):
    return v / linalg.norm(v)