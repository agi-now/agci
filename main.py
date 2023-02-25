import ast

from agci.interpreter import Interpreter
from agci.sst import FunctionEntity, ast_to_sst


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
        'list': list,
        'set': set,
        'len': len,
    })

    functions = []
    for ast_node in ast.parse(CODE).body:
        func_def: ast.FunctionDef = ast_node
        func = FunctionEntity(
            name=func_def.name,
            graph=ast_to_sst.Converter().convert(func_def),
            params=[
                arg.arg for arg in func_def.args.args
            ],
        )
        functions.append(func)
        inr.global_vars[func_def.name] = func

    func = inr.global_vars['main']
    inr.interpret_function(func.graph, func.get_head(), {})


if __name__ == '__main__':
    main()
