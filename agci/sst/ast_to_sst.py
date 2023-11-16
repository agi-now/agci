import re
import ast

from agci import sst


CAMEL_TO_SNAKE = re.compile(r'(?<!^)(?=[A-Z])')
BIN_OP_MAP = {
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.In: 'in',
    ast.NotIn: 'not in',
    ast.Is: 'is',
    ast.IsNot: 'is not',
    ast.Pow: '**',
    ast.BitAnd: '&',
    ast.BitOr: '|',
    ast.BitXor: '^',
}


def _camel_to_snake(name: str):
    return CAMEL_TO_SNAKE.sub('_', name).lower()


class Converter:
    def __init__(self):
        self.nodes: list[sst.Node] = []
        self.edges: list[sst.Edge] = []

        self._last_nodes = []
        self._loop_stack = []

    def _convert(self, ast_node):
        method_name = f"_convert_{_camel_to_snake(type(ast_node).__name__)}"

        if hasattr(self, method_name):
            node = getattr(self, method_name)(ast_node)
            return node
        elif ast_node is None:
            return self._convert_none()
        else:
            print(dir(ast_node))
            print(ast_node)
            breakpoint()
            pass

    def _convert_call(self, ast_node: ast.Call):
        node = sst.FuncCall()
        self.nodes.append(node)

        node_func = self._convert(ast_node.func)

        self.edges.append(sst.Edge(node, node_func, 'func'))

        for i, arg in enumerate(ast_node.args):
            arg_node = self._convert(arg)
            self.edges.append(sst.Edge(node, arg_node, 'args', i))

        for arg in ast_node.keywords:
            arg_node = self._convert(arg.value)
            self.edges.append(sst.Edge(node, arg_node, 'kwargs', arg.arg))

        self._last_nodes = [node, ]

        return node

    def _convert_expr(self, ast_node: ast.Expr):
        return self._convert(ast_node.value)

    def _convert_constant(self, ast_node: ast.Constant):
        node = sst.Constant(value=ast_node.value)
        self.nodes.append(node)

        return node

    def _convert_attribute(self, ast_node: ast.Attribute):
        node = sst.GetAttr(name=ast_node.attr)
        self.nodes.append(node)

        node_value = self._convert(ast_node.value)

        self.edges.append(sst.Edge(node, node_value, 'value'))

        return node

    def _convert_subscript(self, ast_node: ast.Subscript):
        node = sst.GetItem()
        self.nodes.append(node)

        node_value = self._convert(ast_node.value)
        node_slice = self._convert(ast_node.slice)

        self.edges.append(sst.Edge(node, node_value, 'value'))
        self.edges.append(sst.Edge(node, node_slice, 'slice'))

        return node

    def _convert_raise(self, ast_node: ast.Raise):
        node = sst.Raise()
        self.nodes.append(node)

        if ast_node.exc:
            exc_node = self._convert(ast_node.exc)
            self.edges.append(sst.Edge(node, exc_node, 'exc'))

        if ast_node.cause is not None:
            raise SyntaxError("Raise.cause is not supported")

        return node

    def _convert_list(self, ast_node: ast.List):
        node = sst.List()
        self.nodes.append(node)

        for i in range(len(ast_node.elts)):
            elt_node = self._convert(ast_node.elts[i])
            self.edges.append(sst.Edge(node, elt_node, 'elements', i))

        return node

    def _convert_tuple(self, ast_node: ast.Tuple):
        node = sst.Tuple()
        self.nodes.append(node)

        for i in range(len(ast_node.elts)):
            elt_node = self._convert(ast_node.elts[i])
            self.edges.append(sst.Edge(node, elt_node, 'elements', i))

        return node

    def _convert_set(self, ast_node: ast.List):
        node = sst.Set()
        self.nodes.append(node)

        for i in range(len(ast_node.elts)):
            elt_node = self._convert(ast_node.elts[i])
            self.edges.append(sst.Edge(node, elt_node, 'elements'))

        return node

    def __get_assign_targets(self, ast_node):
        if isinstance(ast_node, ast.Name):
            return [ast_node.id, ]
        elif isinstance(ast_node, ast.Tuple):
            return [self.__get_assign_targets(x) for x in ast_node.elts]
        else:
            raise ValueError(f"Invalid target: {ast_node}")

    def _convert_assign(self, ast_node: ast.Assign):
        node = sst.Assign()
        self.nodes.append(node)

        for target in ast_node.targets:
            if isinstance(target, ast.Tuple):
                for elt in target.elts:
                    node_target = self._convert(elt)
                    self.edges.append(sst.Edge(node, node_target, 'targets'))
            else:
                node_target = self._convert(target)
                self.edges.append(sst.Edge(node, node_target, 'targets'))

        node_value = self._convert(ast_node.value)
        self.edges.append(sst.Edge(node, node_value, 'value'))

        self._last_nodes = [node, ]

        return node

    def _convert_aug_assign(self, ast_node: ast.AugAssign):
        node = sst.SelfAssign(op=BIN_OP_MAP[type(ast_node.op)])
        self.nodes.append(node)

        node_target = self._convert(ast_node.target)
        self.edges.append(sst.Edge(node, node_target, 'target'))

        node_value = self._convert(ast_node.value)
        self.edges.append(sst.Edge(node, node_value, 'value'))

        self._last_nodes = [node, ]

        return node

    def _convert_name(self, ast_node: ast.Name):
        node = sst.Variable(name=ast_node.id)
        self.nodes.append(node)
        return node

    def _convert_none(self):
        node = sst.Variable(name='None')
        self.nodes.append(node)
        return node

    def _convert_break(self, ast_node: ast.Break):
        node = sst.Break()
        self.nodes.append(node)

        self._last_nodes = [node, ]

        return node

    def _convert_continue(self, ast_node: ast.Continue):
        node = sst.Continue()
        self.nodes.append(node)

        self._last_nodes = [node, ]

        return node

    def _convert_bool_op(self, ast_node: ast.BoolOp):
        node = sst.BoolOp(op=type(ast_node.op).__name__)

        self.nodes.append(node)

        value_nodes = []
        for i, value in enumerate(ast_node.values):
            value_node = self._convert(value)
            value_nodes.append(value_node)
            self.edges.append(sst.Edge(node, value_node, 'values', i))

        self._last_nodes = [node, ]

        return node

    def _convert_compare(self, ast_node: ast.Compare):
        if len(ast_node.ops) != 1:
            raise ValueError("Expected only one operator")

        op = BIN_OP_MAP[type(ast_node.ops[0])]

        node = sst.Compare(op=op)
        self.nodes.append(node)

        node_left = self._convert(ast_node.left)
        node_right = self._convert(ast_node.comparators[0])

        self.edges.append(sst.Edge(node, node_left, 'left'))
        self.edges.append(sst.Edge(node, node_right, 'right'))

        return node

    def _convert_bin_op(self, ast_node: ast.BinOp):
        op = BIN_OP_MAP[type(ast_node.op)]

        node = sst.BinOp(op=op)
        self.nodes.append(node)

        node_left = self._convert(ast_node.left)
        node_right = self._convert(ast_node.right)

        self.edges.append(sst.Edge(node, node_left, 'left'))
        self.edges.append(sst.Edge(node, node_right, 'right'))

        return node

    def _convert_unary_op(self, ast_node: ast.UnaryOp):
        op = type(ast_node.op).__name__

        node = sst.UnaryOp(op=op)
        self.nodes.append(node)

        node_operand = self._convert(ast_node.operand)
        self.edges.append(sst.Edge(node, node_operand, 'operand'))

        return node

    def _convert_return(self, ast_node: ast.Return):
        node = sst.Return()
        self.nodes.append(node)

        if ast_node.value is not None:
            value_node = self._convert(ast_node.value)
            self.edges.append(sst.Edge(node, value_node, 'value'))

        return node

    def _convert_dict(self, ast_node: ast.Dict):
        node = sst.Dict()
        self.nodes.append(node)

        for i, (key, value) in enumerate(zip(ast_node.keys, ast_node.values)):
            key_node = self._convert(key)
            value_node = self._convert(value)
            self.edges.append(sst.Edge(node, key_node, 'keys', i))
            self.edges.append(sst.Edge(node, value_node, 'values', i))

        return node

    def _convert_if(self, ast_node: ast.If):
        node = sst.If()
        self.nodes.append(node)

        node_test = self._convert(ast_node.test)

        self.edges.append(sst.Edge(node, node_test, 'test'))

        nodes_body = []
        self._last_nodes.clear()

        for sub_ast_node in ast_node.body:
            last_nodes = self._last_nodes.copy()
            sub_node = self._convert(sub_ast_node)
            for last_node in last_nodes:
                self.edges.append(sst.Edge(last_node, sub_node, 'next'))

            nodes_body.append(sub_node)

        nodes_else = []
        self._last_nodes.clear()

        for sub_ast_node in ast_node.orelse:
            last_nodes = self._last_nodes.copy()
            sub_node = self._convert(sub_ast_node)
            for last_node in last_nodes:
                self.edges.append(sst.Edge(last_node, sub_node, 'next'))

            nodes_else.append(sub_node)

        self.edges.append(sst.Edge(node, nodes_body[0], 'next', True))
        if nodes_else:
            self.edges.append(sst.Edge(node, nodes_else[0], 'next', False))

        self._last_nodes = [node, ]

        return node

    def _convert_for(self, ast_node: ast.For):
        if not isinstance(ast_node.target, ast.Name):
            raise ValueError("Only single variables allowed")

        start_node = sst.For(target=ast_node.target.id)
        self.nodes.append(start_node)
        self._loop_stack.append(start_node)

        end_node = sst.EndFor()
        self.nodes.append(end_node)

        iter_node = self._convert(ast_node.iter)
        self.nodes.append(iter_node)

        nodes_body = []
        self._last_nodes.clear()

        for sub_ast_node in ast_node.body:
            last_nodes = self._last_nodes.copy()
            sub_node = self._convert(sub_ast_node)
            for last_node in last_nodes:
                self.edges.append(sst.Edge(last_node, sub_node, 'next'))

            nodes_body.append(sub_node)

        self._loop_stack.pop()

        nodes_else = []
        self._last_nodes.clear()

        for sub_ast_node in ast_node.orelse:
            last_nodes = self._last_nodes.copy()
            sub_node = self._convert(sub_ast_node)
            for last_node in last_nodes:
                self.edges.append(sst.Edge(last_node, sub_node, 'next'))

            nodes_else.append(sub_node)

        self.edges.append(sst.Edge(start_node, end_node, 'next'))
        self.edges.append(sst.Edge(start_node, iter_node, 'iter'))
        self.edges.append(sst.Edge(start_node, nodes_body[0], 'next', True))
        if nodes_else:
            self.edges.append(sst.Edge(start_node, nodes_else[0], 'next', False))

        self._last_nodes = [end_node, ]

        return start_node

    def _convert_while(self, ast_node: ast.While):
        start_node = sst.While()
        self.nodes.append(start_node)
        self._loop_stack.append(start_node)

        end_node = sst.EndWhile()
        self.nodes.append(end_node)

        test_node = self._convert(ast_node.test)
        self.nodes.append(test_node)

        nodes_body = []
        self._last_nodes.clear()

        for sub_ast_node in ast_node.body:
            last_nodes = self._last_nodes.copy()
            sub_node = self._convert(sub_ast_node)
            for last_node in last_nodes:
                self.edges.append(sst.Edge(last_node, sub_node, 'next'))

            nodes_body.append(sub_node)

        self._loop_stack.pop()

        nodes_else = []
        self._last_nodes.clear()

        for sub_ast_node in ast_node.orelse:
            last_nodes = self._last_nodes.copy()
            sub_node = self._convert(sub_ast_node)
            for last_node in last_nodes:
                self.edges.append(sst.Edge(last_node, sub_node, 'next'))

            nodes_else.append(sub_node)

        self.edges.append(sst.Edge(start_node, end_node, 'next'))
        self.edges.append(sst.Edge(start_node, test_node, 'test'))
        self.edges.append(sst.Edge(start_node, nodes_body[0], 'next', True))
        if nodes_else:
            self.edges.append(sst.Edge(start_node, nodes_else[0], 'next', False))

        self._last_nodes = [end_node, ]

        return start_node

    def convert(self, func_def: ast.FunctionDef) -> sst.Graph:
        for ast_node in func_def.body:
            last_nodes = self._last_nodes.copy()
            node = self._convert(ast_node)

            for last_node in last_nodes:
                self.edges.append(sst.Edge(last_node, node, 'next'))

        return sst.Graph(nodes=self.nodes, edges=self.edges)


def convert(ast_graph: ast.FunctionDef) -> sst.Graph:
    cnv = Converter()
    return cnv.convert(ast_graph)
