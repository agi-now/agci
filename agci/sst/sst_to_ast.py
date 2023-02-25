import re
import ast

from agci import sst


CAMEL_TO_SNAKE = re.compile(r'(?<!^)(?=[A-Z])')
BIN_OP_MAP = {
    '>': ast.Gt,
    '>=': ast.GtE,
    '<': ast.Lt,
    '<=': ast.LtE,
    '==': ast.Eq,
    '!=': ast.NotEq,
    '+': ast.Add,
    '-': ast.Sub,
    '*': ast.Mult,
    '/': ast.Div,
}


def _camel_to_snake(name: str):
    return CAMEL_TO_SNAKE.sub('_', name).lower()


class Converter:
    def __init__(self):
        self.nodes: list[sst.Node] = []
        self.edges: list[sst.Edge] = []

        self._last_nodes = []
        self._for_stack = []

    def _convert(self, graph: sst.Graph, sst_node):
        method_name = f"_convert_{_camel_to_snake(type(sst_node).__name__)}"

        if hasattr(self, method_name):
            node = getattr(self, method_name)(graph, sst_node)
            return node
        else:
            print(sst_node)
            breakpoint()
            pass

    def _convert_assign(self, graph: sst.Graph, sst_node: sst.Assign):
        ast_node = ast.Assign()
        targets = graph.out(sst_node, 'targets')
        if len(targets) > 1:
            ast_node.targets = [
                ast.Tuple(elts=[
                    self._convert(graph, target)
                    for target in targets
                ])
            ]
        else:
            ast_node.targets = [
                self._convert(graph, targets[0])
            ]
        ast_node.value = self._convert(graph, graph.out_one(sst_node, 'value'))

        return ast_node

    def _convert_bin_op(self, graph: sst.Graph, sst_node: sst.BinOp):
        ast_node = ast.BinOp()
        ast_node.op = BIN_OP_MAP[sst_node.op]()
        ast_node.left = self._convert(graph, graph.out_one(sst_node, 'left'))
        ast_node.right = self._convert(graph, graph.out_one(sst_node, 'right'))

        return ast_node

    def _convert_unary_op(self, graph: sst.Graph, sst_node: sst.UnaryOp):
        ast_node = ast.UnaryOp()
        ast_node.op = getattr(ast, sst_node.op)()
        ast_node.operand = self._convert(graph, graph.out_one(sst_node, 'operand'))

        return ast_node

    def _convert_compare(self, graph: sst.Graph, sst_node: sst.Compare):
        ast_node = ast.Compare()
        ast_node.ops = [BIN_OP_MAP[sst_node.op]()]
        ast_node.left = self._convert(graph, graph.out_one(sst_node, 'left'))
        ast_node.comparators = [
            self._convert(graph, graph.out_one(sst_node, 'right'))
        ]
        return ast_node

    def _convert_break(self, graph: sst.Graph, sst_node: sst.Break):
        ast_node = ast.Break()
        return ast_node

    def _convert_dict(self, graph: sst.Graph, sst_node: sst.Dict):
        ast_node = ast.Dict()
        ast_node.keys = []
        ast_node.values = []

        keys = graph.out_edges(sst_node, 'keys')
        keys.sort(key=lambda x: x.param)

        values = graph.out_edges(sst_node, 'values')
        values.sort(key=lambda x: x.param)

        for key, value in zip(keys, values):
            assert key.param == value.param
            ast_node.keys.append(self._convert(graph, key.end))
            ast_node.values.append(self._convert(graph, value.end))

        return ast_node

    def _convert_list(self, graph: sst.Graph, sst_node: sst.List):
        ast_node = ast.List()
        ast_node.elts = []

        elements = graph.out_edges(sst_node, 'elements')
        elements.sort(key=lambda x: x.param)

        for element in elements:
            ast_node.elts.append(self._convert(graph, element.end))

        return ast_node

    def _convert_return(self, graph: sst.Graph, sst_node: sst.Return):
        ast_node = ast.Return()
        ast_node.value = self._convert(graph, graph.out_one(sst_node, 'value'))

        return ast_node

    def _convert_self_assign(self, graph: sst.Graph, sst_node: sst.SelfAssign):
        ast_node = ast.AugAssign()
        ast_node.target = self._convert(graph, graph.out_one(sst_node, 'target'))
        ast_node.value = self._convert(graph, graph.out_one(sst_node, 'value'))
        ast_node.op = BIN_OP_MAP[sst_node.op]()

        return ast_node

    def _convert_get_attr(self, graph: sst.Graph, sst_node: sst.GetAttr):
        ast_node = ast.Attribute()
        ast_node.attr = sst_node.name
        ast_node.value = self._convert(graph, graph.out_one(sst_node, 'value'))

        return ast_node

    def _convert_get_item(self, graph: sst.Graph, sst_node: sst.GetItem):
        ast_node = ast.Subscript()
        ast_node.slice = self._convert(graph, graph.out_one(sst_node, 'slice'))
        ast_node.value = self._convert(graph, graph.out_one(sst_node, 'value'))

        return ast_node

    def _convert_if(self, graph: sst.Graph, sst_node: sst.If):
        ast_node = ast.If()
        ast_node.test = self._convert(graph, graph.out_one(sst_node, 'test'))
        ast_node.body = self._convert_body(graph, graph.out_one(sst_node, 'next', True))
        out_false = graph.out(sst_node, 'next', False)
        if len(out_false) > 1:
            raise ValueError()

        if len(out_false) == 1:
            ast_node.orelse = self._convert_body(graph, out_false[0])
        else:
            ast_node.orelse = []

        return ast_node

    def _convert_for(self, graph: sst.Graph, sst_node: sst.For):
        ast_node = ast.For()

        ast_node.iter = self._convert(graph, graph.out_one(sst_node, 'iter'))
        ast_node.target = ast.Name(id=sst_node.target)

        ast_node.body = self._convert_body(graph, graph.out_one(sst_node, 'next', True))
        out_false = graph.out(sst_node, 'next', False)
        if len(out_false) > 1:
            raise ValueError()

        if len(out_false) == 1:
            ast_node.orelse = self._convert_body(graph, out_false[0])
        else:
            ast_node.orelse = []

        return ast_node

    def _convert_while(self, graph: sst.Graph, sst_node: sst.For):
        ast_node = ast.While()

        ast_node.test = self._convert(graph, graph.out_one(sst_node, 'test'))

        ast_node.body = self._convert_body(graph, graph.out_one(sst_node, 'next', True))
        out_false = graph.out(sst_node, 'next', False)
        if len(out_false) > 1:
            raise ValueError()

        if len(out_false) == 1:
            ast_node.orelse = self._convert_body(graph, out_false[0])
        else:
            ast_node.orelse = []

        return ast_node

    def _convert_variable(self, graph: sst.Graph, sst_node: sst.Variable):
        ast_node = ast.Name(id=sst_node.name)
        return ast_node

    def _convert_func_call(self, graph: sst.Graph, sst_node: sst.FuncCall):
        ast_node = ast.Call()
        ast_node.func = self._convert(graph, graph.out_one(sst_node, 'func'))
        args = graph.out_edges(sst_node, 'args')
        args.sort(key=lambda x: x.param)
        kwargs = graph.out_edges(sst_node, 'kwargs')
        ast_node.args = [self._convert(graph, arg.end) for arg in args]
        ast_node.keywords = [
            ast.keyword(arg=kwarg.param, value=self._convert(graph, kwarg.end)) for kwarg in kwargs]

        return ast_node

    def _convert_constant(self, graph: sst.Graph, sst_node: sst.Constant):
        return ast.Constant(value=sst_node.value)

    def _convert_body(self, graph, first_node):
        body = []

        while True:
            if not isinstance(first_node, sst.EndFor) and not isinstance(first_node, sst.EndWhile):
                ast_node = self._convert(graph, first_node)
                if isinstance(ast_node, ast.Call):
                    # wrap in Expr
                    ast_node = ast.Expr(value=ast_node)
                body.append(ast_node)

            next_nodes = graph.out(first_node, 'next')
            if len(next_nodes) == 0:
                break
            else:
                first_node = next_nodes[0]

        return body

    def convert(self, graph: sst.Graph) -> ast.FunctionDef:
        body = self._convert_body(graph, graph.get_nodes()[0])
        func_def = ast.FunctionDef(body=body, name='test', decorator_list=[], args=[])

        return ast.fix_missing_locations(func_def)


def main():
    g = sst.Graph([
        sst.Assign(),
        sst.Variable('abc'),
        sst.Variable('test'),

        sst.For('x'),
        sst.Variable('samp'),

        sst.If(),
        sst.Variable('test'),

        sst.Assign(),
        sst.Variable('test'),
        sst.Variable('abc'),

        sst.Assign(),
        sst.Variable('test2'),
        sst.Variable('abc2'),

        sst.EndFor(),

        sst.Assign(),
        sst.Variable('test3'),
        sst.Constant(value=12),

    ], [])

    g.edges = [
        sst.Edge(g.nodes[0], g.nodes[1], 'targets'),
        sst.Edge(g.nodes[0], g.nodes[2], 'value'),

        sst.Edge(g.nodes[0], g.nodes[3], 'next'),

        sst.Edge(g.nodes[3], g.nodes[4], 'iter'),
        sst.Edge(g.nodes[3], g.nodes[5], 'next', True),
        # sst.Edge(g.nodes[3], g.nodes[13], 'next', False),
        sst.Edge(g.nodes[3], g.nodes[13], 'next'),

        sst.Edge(g.nodes[5], g.nodes[6], 'test'),

        sst.Edge(g.nodes[5], g.nodes[7], 'next', True),
        sst.Edge(g.nodes[5], g.nodes[10], 'next'),

        sst.Edge(g.nodes[7], g.nodes[8], 'targets'),
        sst.Edge(g.nodes[7], g.nodes[9], 'value'),

        sst.Edge(g.nodes[10], g.nodes[11], 'targets'),
        sst.Edge(g.nodes[10], g.nodes[12], 'value'),

        sst.Edge(g.nodes[13], g.nodes[14], 'next'),

        sst.Edge(g.nodes[14], g.nodes[15], 'targets'),
        sst.Edge(g.nodes[14], g.nodes[16], 'value'),
    ]

    cnv = Converter()
    ast_func_def = cnv.convert(g)
    ast_func_def = ast.fix_missing_locations(ast_func_def)

    print(ast.unparse(ast_func_def))
    print()

    breakpoint()
    pass


if __name__ == '__main__':
    main()

