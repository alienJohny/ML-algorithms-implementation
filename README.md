# ML Algorithms Implementation

## Градиентный спуск
В общем случае имеется некоторая целевая функция `target_fn`, которую требуется минимизировать,<br />
```text
arg min target_fn
```
а также минимизировать ее градиент `gradient_fn`. <br />
К примеру, такой градиент `gradient_fn` может представлять ошибки в модели, как функцию своих параметров, <br />
и тогда мы могли бы отыскать параметры, которые минимизируют ошибки насколько, насколько это возможно.<br />
<br />
Далее, пусть каким то образом выбрано исходное значение параметров `theta_0`.<br />
Тогда метод градиентного спуска реализуется следующим образом:<br />
<br />

```python
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

```

Функция названа пакетной минимизацией, потому что на каждой итерации считается ошибка на всем наборе данных.
<br />

## Стохастический градиентный спуск
Как уже упоминалось ранее, метод градиентого спуска, как правило, будет применяться для выбора параметров <br />
модели таким образом, чтобы минимизировать некое представление об ошибке.<br />
<br />
Функции ошибок как правило имеют свойство аддитивности, то есть состоят из вычисления ошибки на каждом объекте.<br />
<br />
В этом случае можно применить `метод стохастического градиентого спуска` (SGD), который за одну итерацию<br />
вычисляет градиент и делает шаг только на одной точке. Он многократно просматривает данные, пока не достигнет схождения.<br />
Во время цикла просмотр данных выполняется в случайном порядке.
```python
def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    data = zip(x, y)
    theta = theta_0
    alpha = alpha_0 # Step size
    min_theta, min_value = None, float('inf')
    iterations_with_no_improvement = 0

    # Stop if you have reached 100 iterations without improvement
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if (value < min_value):
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9

            # Make a gradient step for each data point
            for x_i, y_i in in_random_order(data):
                gradient_i = gradient_fn(x_i, y_i, theta)
                theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta
```
