# Agent Code Interpreter

## About
Library to convert python code to traversable [AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree) and back to code.  
By traversable AST we mean AST where every node has "next" edge pointing to the next node to execute.  
Branching nodes have separate "next" edges for True and False condition.  
Regular python AST is not actually represented as graph, so we need to do this converions:  
    `Python -> Python AST -> Traversable AST.`  
Traversible AST is then being interpreted by the interepter.  
We need AST to be traversable in order to do pattern matching on it, traversability is not a requirement for running the code.  

## Interpreter Usage

```python
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

```

## Conversion Usage
```python
from agci.sst import ast_to_sst
from agci.sst import sst_to_ast

ast_graph = ast.parse(CODE)
traversable_graph = ast_to_sst.convert(ast_graph.body[0])

# And convert back
ast_graph = sst_to_ast.convert(traversable_graph)
code = ast.unparse(ast_graph)

```
