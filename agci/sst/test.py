import ast

import sst
import sst.ast_to_sst
import sst.sst_to_ast


def main():
    code = """

def test():
    g += a
    g += a.b
    g += a.b()
    g.b += a.b()
    if test:
        print.te()
        print.me()
    for a in b:
        call1()
        call2().call2()
        for a2 in ctx[run()[3]].b2():
            call3().call3().call3()
            call4().call4().call4().call4()
    print(123)


def test():
    abc = ctx.kb.load()
    ctx.my = 23

    for x in abc:
        start, stop = x
        if start[0] > stop:
            break
        else:
            ctx.start += 1
            print(stop)

    print('end')
    """

    func_def: ast.FunctionDef = ast.parse(code).body[1]

    graph = sst.ast_to_sst.Converter().convert(func_def)
    func_def = sst.sst_to_ast.Converter().convert(graph)

    print(ast.unparse(func_def))


if __name__ == '__main__':
    main()
