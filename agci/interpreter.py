import ast
import operator
from dataclasses import dataclass
from typing import Optional

from agci import sst
from agci.sst import ast_to_sst
from agci.sst import Graph, FunctionEntity


class NoReturnValue:
    pass


NO_RETURN_VALUE = NoReturnValue()

BIN_OP_MAP = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
    'in': operator.contains,
    'is': operator.is_,
    'is not': operator.is_not,
    '**': operator.pow,
    '&': operator.and_,
    '|': operator.or_,
    '^': operator.xor,
}


@dataclass
class InterpreterContext:
    variables: dict[str, any]
    return_value = NO_RETURN_VALUE

    def get(self, name):
        try:
            return self.variables[name]
        except KeyError as e:
            raise KeyError(f'Variable "{name}" not found!') from e


class Interpreter:
    def __init__(self, global_vars):
        self.global_vars = global_vars
        self.ctx = []
        self.break_set = False
        self.continue_set = False

    def _interpret_func_call(self, graph, node):
        func, _ = self.interpret_node(graph, graph.out_one(node, 'func'))
        args = []
        kwargs = {}
        for arg_edge in graph.out_edges(node, 'args'):
            args.append(self.interpret_node(graph, arg_edge.end)[0])

        for kwarg_edge in graph.out_edges(node, 'kwargs'):
            if kwarg_edge.param is None:
                raise ValueError('kwargs must be named')
            kwargs[kwarg_edge.param] = self.interpret_node(graph, kwarg_edge.end)[0]

        if isinstance(func, FunctionEntity):
            params = {}
            not_filled_params = func.params.copy()
            for key in kwargs:
                params[key] = kwargs[key]
                not_filled_params.remove(key)
            for i, arg in enumerate(args):
                params[not_filled_params[i]] = arg

            return self.interpret_function(func.graph, func.get_head(), params), \
                graph.out_one(node, 'next', optional=True)

        return func(*args, **kwargs), graph.out_one(node, 'next', optional=True)

    def _interpret_if(self, graph, node):
        test_node = graph.out_one(node, 'test')
        true_head = graph.out_one(node, 'next', True)
        false_head = graph.out_one(node, 'next', False, optional=True)
        test, _ = self.interpret_node(graph, test_node)
        if test:
            self.interpret_body(graph, true_head)
        elif false_head is not None:
            self.interpret_body(graph, false_head)
        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_while(self, graph, node):
        self.break_set = self.continue_set = False

        test_node = graph.out_one(node, 'test')
        true_head = graph.out_one(node, 'next', True)
        # else_head = graph.out_one(node, 'next', False, optional=True)
        test, _ = self.interpret_node(graph, test_node)

        while test:
            self.interpret_body(graph, true_head)
            test, _ = self.interpret_node(graph, test_node)
            self.continue_set = False
            if self.break_set:
                self.break_set = False
                break
            if self.ctx[-1].return_value is not NO_RETURN_VALUE:
                break

        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_for(self, graph, node: sst.For):
        self.break_set = self.continue_set = False

        iter_node = graph.out_one(node, 'iter')
        true_head = graph.out_one(node, 'next', True)
        # else_head = graph.out_one(node, 'next', False, optional=True)
        iterator, _ = self.interpret_node(graph, iter_node)

        for val in iterator:
            self.ctx[-1].variables[node.target] = val
            self.interpret_body(graph, true_head)
            self.continue_set = False
            if self.break_set:
                self.break_set = False
                break
            if self.ctx[-1].return_value is not NO_RETURN_VALUE:
                break

        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_assign(self, graph, node):
        targets = graph.out(node, 'targets')
        value, _ = self.interpret_node(graph, graph.out_one(node, 'value'))
        for target in targets:
            assert isinstance(target, sst.Variable)
            self.ctx[-1].variables[target.name] = value

        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_self_assign(self, graph, node):
        value, _ = self.interpret_node(graph, graph.out_one(node, 'value'))
        target = graph.out_one(node, 'target')
        assert isinstance(target, sst.Variable)
        if node.op == '+':
            self.ctx[-1].variables[target.name] += value
        elif node.op == '-':
            self.ctx[-1].variables[target.name] -= value
        elif node.op == '*':
            self.ctx[-1].variables[target.name] *= value
        elif node.op == '/':
            self.ctx[-1].variables[target.name] /= value
        else:
            raise ValueError(node.op)

        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_bin_op(self, graph, node):
        left, right = (
            self.interpret_node(graph, graph.out_one(node, 'left'))[0],
            self.interpret_node(graph, graph.out_one(node, 'right'))[0],
        )
        if node.op == 'in':
            left, right = right, left

        result = BIN_OP_MAP[node.op](
            left, right
        )

        return result, graph.out_one(node, 'next', optional=True)

    def _interpret_bool_op(self, graph, node):
        value_edges = graph.out_edges(node, 'values')
        value_edges.sort(key=lambda x: x.param)
        values = [
            self.interpret_node(graph, edge.end)[0]
            for edge in value_edges
        ]
        if node.op == 'And':
            result = True
            for value in values:
                result = result and value
        elif node.op == 'Or':
            result = False
            for value in values:
                result = result or value
        else:
            raise ValueError(node.op)

        return result, graph.out_one(node, 'next', optional=True)

    def _interpret_unary_op(self, graph, node: sst.UnaryOp):
        operand = self.interpret_node(graph, graph.out_one(node, 'operand'))[0]

        if node.op == 'USub':
            result = -operand
        else:
            raise ValueError(node.op)

        return result, graph.out_one(node, 'next', optional=True)

    def interpret_node(self, graph: Graph, node):
        if isinstance(node, sst.FuncCall):
            return self._interpret_func_call(graph, node)

        elif isinstance(node, sst.If):
            return self._interpret_if(graph, node)

        elif isinstance(node, sst.While):
            return self._interpret_while(graph, node)

        elif isinstance(node, sst.For):
            return self._interpret_for(graph, node)

        elif isinstance(node, sst.Assign):
            return self._interpret_assign(graph, node)

        elif isinstance(node, sst.SelfAssign):
            return self._interpret_self_assign(graph, node)

        elif isinstance(node, sst.BinOp):
            return self._interpret_bin_op(graph, node)

        elif isinstance(node, sst.Compare):
            return self._interpret_bin_op(graph, node)

        elif isinstance(node, sst.BoolOp):
            return self._interpret_bool_op(graph, node)

        elif isinstance(node, sst.UnaryOp):
            return self._interpret_unary_op(graph, node)

        elif isinstance(node, sst.Constant):
            return node.value, graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.List):
            value = []
            elements = graph.out_edges(node, 'elements')
            for edge in elements:
                value.append(self.interpret_node(graph, edge.end)[0])
            return value, graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.Dict):
            result = {}
            keys = graph.out_edges(node, 'keys')
            values = graph.out_edges(node, 'values')
            for key, value in zip(keys, values):
                assert key.param == value.param

                result[self.interpret_node(graph, key.end)[0]] =\
                    self.interpret_node(graph, value.end)[0]

            return result, graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.Variable):
            return self.ctx[-1].get(node.name), graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.GetAttr):
            value = self.interpret_node(graph, graph.out_one(node, 'value'))[0]
            return getattr(value, node.name), graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.GetItem):
            value = self.interpret_node(graph, graph.out_one(node, 'value'))[0]
            _slice = self.interpret_node(graph, graph.out_one(node, 'slice'))[0]
            return value[_slice], graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.Return):
            self.ctx[-1].return_value = self.interpret_node(graph, graph.out_one(node, 'value'))[0]
            return None, None

        elif isinstance(node, sst.Break):
            self.break_set = True
            return None, None
        
        elif isinstance(node, sst.Continue):
            self.continue_set = True
            return None, None

        elif isinstance(node, sst.EndWhile):
            self.continue_set = False
            self.break_set = False
            return None, graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.EndFor):
            self.continue_set = False
            self.break_set = False
            return None, graph.out_one(node, 'next', optional=True)

        else:
            raise ValueError(node)

    def interpret_body(self, graph: Graph, head):
        assert head is not None
        result = None

        while head is not None:
            result, head = self.interpret_node(graph, head)
            if self.ctx[-1].return_value is not NO_RETURN_VALUE:
                return
            if self.continue_set or self.break_set:
                return

        return result

    def interpret_function(self, graph: Graph, head, kwargs: dict):
        # self.code_history.add_tree(graph)
        self.ctx.append(InterpreterContext(
            variables={
                **self.global_vars,
                **kwargs.copy(),
            }
        ))
        self.interpret_body(graph, head)
        ctx = self.ctx.pop()

        if ctx.return_value is NO_RETURN_VALUE:
            return None
        return ctx.return_value

    def load_code(self, code):
        for ast_node in ast.parse(code).body:
            func_def: ast.FunctionDef = ast_node
            func = FunctionEntity(
                name=func_def.name,
                graph=ast_to_sst.Converter().convert(func_def),
                params=[
                    arg.arg for arg in func_def.args.args
                ],
            )
            self.global_vars[func_def.name] = func

    def run_main(self):
        func = self.global_vars['main']
        return self.interpret_function(func.graph, func.get_head(), {})
