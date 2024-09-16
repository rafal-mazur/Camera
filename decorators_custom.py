from time import time

def measure_exe_time(func):
    def wrap(*args, **kwargs):
        start = time()
        func_result = func(*args, **kwargs)
        end = time()
        print(f'Execution time of {func.__name__}: {end - start} s')
        return func_result

    return wrap
