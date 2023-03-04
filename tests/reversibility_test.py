import ast

import pytest

import agci.sst.ast_to_sst
import agci.sst.sst_to_ast


@pytest.mark.parametrize("code", [
    """
def test():
    abc = ctx.first.second(var1, test=1 + 1)
    x = not x
    x = -x
    x = ~x
    x = +x
""", """
def test():
    ctx.first.second(var1, test=1 + 1)
""", """
def test():
    ctx.first.second(var1, test=1 + 1)
""", """
def test():
    ctx[1].first.second[0](var1, test=1 + 1)
""", """
def test():
    abc = {'test': call(call2(), arg1=1)}
""", """
def test():
    abc = {1 + 1: call(call2(), arg1=1)}
""", """
def test():
    abc = {1 + 1: [1, 2]}
""", """
def test():
    run(test=[call()])
""", """
def test():
    return '1' == '2'
""", """
def test():
    return '1' != '2'
""", """
def test():
    return '1' >= '2'
""", """
def test():
    return '1' <= '2'
""", """
def test():
    return '1' > '2'
""", """
def test():
    return '1' < '2'
""", """
def test():
    return pow(**temp)
""", """
def test():
    if len([]) == 0:
        run('test')
        ctx.test(8 + 9)
    run('test' + '9')
""", """
def test():
    if len([]) == 0:
        run('test')
        ctx.test(8 + 9)
    elif a > b:
        return 123
    elif b > a:
        for _ in range(c):
            break
    else:
        return
    run('test' + '9')
""", """
def test():
    if len([]) == 0:
        run('test')
        ctx.test(8 + 9)
    else:
        ctx.call[-1] = 2
    run('test' + '9')
""", """
def test():
    for test in [1, 2, call(), {'': 1}]:
        run(test)
        ctx.test(8 + 9)
    else:
        ctx.call[-1] = 2
    run('test' + '9')
""", """
def test():
    while test:
        run(test)
        ctx.test(8 + 9)
    else:
        ctx.call[-1] = 2
    run('test' + '9')
""", """
def test():
    while test and True:
        run(test)
        for test in [1, 2, call(), {'': 1}]:
            run(test)
            ctx.test(8 + 9)
            continue
        else:
            break
            ctx.call[-1] = 2
            call(2)
        ctx.test(8 + 9)
    else:
        ctx.call[-1] = 2
        for test in [1, 2, call(), {'': 1}]:
            ctx.test(8 + 9)
            run(test)
        else:
            call(7)
            ctx.call[-1] = 2
        print('3')
    run('test' + '9')
"""
])
def test_reversibility(code):
    func_def1: ast.FunctionDef = ast.parse(code).body[0]
    graph = agci.sst.ast_to_sst.Converter().convert(func_def1)
    func_def2 = agci.sst.sst_to_ast.Converter().convert(graph)

    out_code = ast.unparse(func_def2)
    assert code.strip() == out_code.strip()

