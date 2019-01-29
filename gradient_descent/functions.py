def difference_quotient(f, x, dx):
    return (f(x + dx) - f(x)) / dx

def partial_difference_quotient(f, v, i, h):
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]

# make a gradient step
def step(v, direction, step_size):
    return [v_i + step_size * direction_i for v_i, direction_i in zip(v, direction)]

def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f


def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=1e-8):
    """
    Uses gradient descent algorithm for minimizing a target function `target_fn`

    :param target_fn: Callable, target function
    :param gradient_fn: Vector of partial derivatives
    :param theta_0: Vector of parameters
    :param tolerance: Accuracy
    :return: Vector of new parameters theta, theta.shape = theta_0.shape
    """

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0 # Set theta to the initial value
    target_fn = safe(target_fn)

    value = target_fn(theta)

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]

        # Choose gradient, which minimize error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # Stop if convergence
        if (abs(value - next_value) < tolerance):
            return theta
        else:
            theta, value = next_theta, next_value

    