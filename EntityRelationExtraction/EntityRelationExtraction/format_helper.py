def printHeader(string):
    print("*" * 100)
    print(f"{string.upper().center(100, '-')}")
    print("*" * 100)
    print()


def printEnd(string):
    print()
    print("*" * 100)
    string = "Finished " + string
    print(f"{string.upper().center(100, '-')}")
    print("*" * 100)
    print("\n" * 3)


def ProgressDecorator(string):
    def real_decorator(function):
        def wrapper(*args):
            printHeader(string)
            res = function(*args)
            printEnd(string)
            return res
        return wrapper
    return real_decorator
