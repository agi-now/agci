# Agent Code Interpreter

## About

Library to convert python code to [Control Flow Graph](https://en.wikipedia.org/wiki/Control-flow_graph) and back to code.  

The conversion is done with an intermediary step of converting to an [Abstract Syntax Tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree):  
    `Python -> AST -> CFG.`  
CFG is then being interpreted.  
We need CFG because we want to do pattern matching on the code while it's being executed.  

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

inr = Interpreter(global_vars={
    'print': print,
})

inr.load_code(CODE)
list(inr.run_main())
```

## Conversion Usage

```python
import ast

from agci.sst import ast_to_sst
from agci.sst import sst_to_ast

CODE = """
def main():
    print(factorial(7))
    

def factorial(x):
    if x == 1:
        return x
    return x * factorial(x - 1)
"""

ast_graph = ast.parse(CODE)
cfg_graph = ast_to_sst.convert(ast_graph.body[0])

# And convert back
ast_graph = sst_to_ast.convert(cfg_graph)
code = ast.unparse(ast_graph)
```
