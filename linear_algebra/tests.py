from termcolor import colored
from functions import *

def test_f(f, test_v, expected_v):
    f_name = f.__name__ + " " * (20 - len(f.__name__))
    if (test_v == expected_v):
        print(colored(f_name + " : tested --- OK", "green"))
    else:
        print(colored(f_name + " : tested --- ERROR", "red"))

def unit_tests():
    test_f(vector_add, vector_add([1, 2], [2, 1]), [3, 3])
    test_f(vector_subtract, vector_subtract([1, 2], [2, 1]), [-1, 1])
    test_f(vector_sum, vector_sum([[1], [1], [1]]), [3])
    test_f(scalar_multiply, scalar_multiply(2, [1, -1]), [2, -2])
    test_f(dot, dot([1, 2, 3], [1, 2, 3]), 14)
    test_f(sum_of_squares, sum_of_squares([2, -2]), 8)
    test_f(magnitude, magnitude([1, 1, 1, 1]), 2)
    test_f(squared_distance, squared_distance([1, 2], [2, 1]), 2)
    test_f(distance, distance([1, 0], [3, 0]), 2)
    test_f(shape, shape([[0]]), (1, 1))

def main():
    unit_tests()

if (__name__ == "__main__"):
    main()
