import math

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vs):
    result = vs[0]
    for v in vs[1:]:
        result = vector_add(result, v)
    return result

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    return dot(v, v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
    # Euclidian distance
    return magnitude(vector_subtract(v, w))

def shape(A):
    m = len(A)
    n = len(A[0]) if A else 0
    return (m, n)


