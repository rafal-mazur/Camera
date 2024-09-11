from time import time


def measure_exe_time(func):
    def wrap(*args, **kwargs):
        start = time()
        func_result = func(*args, **kwargs)
        end = time()
        print(f'Execution time of {func.__name__}: {end - start} s')
        return func_result

    return wrap

# TODO: znaleźć sposob na dodanie parametrów do dekoratorów
def repeat(func, n: int = 3):
    def wrap(*args, **kwargs) -> None:
        for i in range(n):
            print(f'Run {i} out of {n}')
            func(*args, **kwargs)
            print()

    return wrap
