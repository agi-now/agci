# Agent Code Interpreter

## About
Library to convert python code to traversible [AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree) and back to code.  
By traversible AST we mean AST where every node has "next" edge pointing to the next node to execute.
Branching nodes have separate "next" edges for True and False condition.
Regular python AST is not actually represented as graph, so we need to do this converions:
Python -> Python AST -> Traversible AST.
Traversible AST is then being interpreted by the interepter.  
We need AST to be traversible in order to do pattern matching on it, traversability is not a requirement for running the code.

## Usage

```
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
    inr = Interpreter({
        'print': print,
    })

    inr.load_code(CODE)
    inr.run_main()


if __name__ == '__main__':
    main()

```
