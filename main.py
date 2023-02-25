from agci import Interpreter

CODE = """
def main():
    print(factorial(7))
    

def factorial(x):
    if x == 1:
        return x
    return x * factorial(x - 1)
"""


def main():
    inr = Interpreter(global_vars={
        'print': print,
    })

    inr.load_code(CODE)
    inr.run_main()


if __name__ == '__main__':
    main()
