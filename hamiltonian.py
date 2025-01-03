import numpy as np

def first_finite_diff(f, x, delta):
    """
    Compute the first derivative of a function f using the finite difference method.

    Parameters:
        f (function): Function to differentiate.
        x (float or array): Point(s) at which to evaluate the first derivative.
        delta (float): Small step size for finite difference.

    Returns:
        array: First derivative of f at x.
    """
    derivative_first = (f(x + delta) - f(x - delta)) / (delta ** 2)

    return derivative_first

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
    derivative_second = (f(x + delta) - 2 * f(x) + f(x - delta)) / (delta ** 2)

    return derivative_second




