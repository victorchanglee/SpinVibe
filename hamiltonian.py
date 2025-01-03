import numpy as np

def first_finite_diff(f, x, delta):
    """
    Compute the first derivative of a function f using the finite difference method.

    Parameters:
        f (function): Function to differentiate.
        x (array): Coordinates at which to evaluate the derivative.
        delta (float): Small step size for finite difference.

    Returns:
        array: First derivative of f at x.
    """
    derivatives = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += delta
        x_backward[i] -= delta
        derivatives[i] = (f(x_forward) - f(x_backward)) / (2 * delta)
    return derivatives

def second_finite_diff(f,x,delta):
    """
    Compute the second derivative of a function f using the finite difference method.

    Parameters:
        f (function): Function to differentiate.
        x (float or array): Point(s) at which to evaluate the second derivative.
        delta (float): Small step size for finite difference.

    Returns:
        float or array: Second derivative of f at x.
    """
    finite_second = (f(x + delta) - 2 * f(x) + f(x - delta)) / (delta ** 2)

    return finite_second




