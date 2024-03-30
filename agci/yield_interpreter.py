import ast

from agci.interpreter import (
    BIN_OP_MAP, 
    NO_RETURN_VALUE, 
    InterpreterContext,
)
from agci.sst.entities import FunctionDispatchOption

from . import sst
from .sst import Graph, FunctionEntity
from .sst import ast_to_sst


def _args_hash_func(args):
    return '-'.join([
        f'{name}={concept}'
        for name, concept in args
    ])


def _convert_ast_node_to_concept(ast_node):
    if ast_node is None:
        return None
    
    if isinstance(ast_node, ast.Name):
        return ast_node.id
    
    if isinstance(ast_node, ast.Call):
        name = ast_node.func.id
        assert ast_node.args == []
        fields = {
            kw.arg: _convert_ast_node_to_concept(kw.value)
            for kw in ast_node.keywords
        }
        
        cid = [name, ]

        if fields:
            cid.append('{')
            for key, value in sorted(fields.items(), key=lambda x: x[0]):
                cid.append(f'{key}={value},')
            cid[-1] = cid[-1][:-1]
            cid.append('}')

        return ''.join(cid)
    
    raise NotImplementedError(ast_node)


class YieldaInterpreter:
    def __init__(self, global_vars: dict[str, any]):
        global_vars['__agci'] = self
        self.global_vars = global_vars
        self.ctx = []
        self.break_set = False
        self.continue_set = False
        self.combined_graph = Graph([], [])
        self.func_defs = []

    def _interpret_func_call(self, graph, node):
        func, _ = yield from self.interpret_node(graph, graph.out_one(node, 'func'))
        args = []
        kwargs = {}
        
        for arg_edge in graph.out_edges(node, 'args'):
            arg_value, _ = yield from self.interpret_node(graph, arg_edge.end)
            args.append(arg_value)

        for kwarg_edge in graph.out_edges(node, 'kwargs'):
            kwarg_value, _ = yield from self.interpret_node(graph, kwarg_edge.end)
            if kwarg_edge.param is None:
                kwargs.update(kwarg_value)
            else:
                kwargs[kwarg_edge.param] = kwarg_value
                
        if isinstance(func, FunctionEntity):
            # can't be interpreted with __call__ because it needs to be merged with the current graph
            # and then yielded from
            
            func_graph, new_kwargs = func.resolve_dispatch(args, kwargs)
            # graph.merge(func_graph, node, edge='func_code')
            
            value = yield from func(**new_kwargs)
            
            return value, graph.out_one(node, 'next', optional=True)

        return func(*args, **kwargs), graph.out_one(node, 'next', optional=True)

    def _interpret_if(self, graph, node):
        test_node = graph.out_one(node, 'test')
        true_head = graph.out_one(node, 'next', True)
        false_head = graph.out_one(node, 'next', False, optional=True)
        test, _ = yield from self.interpret_node(graph, test_node)
        if test:
            yield from self.interpret_body(graph, true_head)
        elif false_head is not None:
            yield from self.interpret_body(graph, false_head)
        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_while(self, graph, node):
        self.break_set = self.continue_set = False

        test_node = graph.out_one(node, 'test')
        true_head = graph.out_one(node, 'next', True)
        # else_head = graph.out_one(node, 'next', False, optional=True)
        test, _ = yield from self.interpret_node(graph, test_node)

        while test:
            yield from self.interpret_body(graph, true_head)
            test, _ = yield from self.interpret_node(graph, test_node)
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
        iterator, _ = yield from self.interpret_node(graph, iter_node)

        for val in iterator:
            self.ctx[-1].variables[node.target] = val
            yield from self.interpret_body(graph, true_head)
            self.continue_set = False
            if self.break_set:
                self.break_set = False
                break
            if self.ctx[-1].return_value is not NO_RETURN_VALUE:
                break

        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_assign(self, graph, node):
        targets = graph.out(node, 'targets')
        value, _ = yield from self.interpret_node(graph, graph.out_one(node, 'value'))
        for target in targets:
            if isinstance(target, sst.Variable):
                graph.active_node_id = target.node_id
                yield
                self.ctx[-1].variables[target.name] = value
            elif isinstance(target, sst.GetItem):
                target_value, _ = yield from self.interpret_node(graph, graph.out_one(target, 'value'))
                _slice, _ = yield from self.interpret_node(graph, graph.out_one(target, 'slice'))
                target_value[_slice] = value
            elif isinstance(target, sst.GetAttr):
                target_value, _ = yield from self.interpret_node(graph, graph.out_one(target, 'value'))
                setattr(target_value, target.name, value)
            else:
                raise ValueError()

        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_self_assign(self, graph, node):
        value, _ = yield from self.interpret_node(graph, graph.out_one(node, 'value'))
        target = graph.out_one(node, 'target')
        
        if isinstance(target, sst.Variable):
            target_upper = self.ctx[-1].variables
            field = target.name
        elif isinstance(target, sst.GetItem):
            target_upper, _ = yield from self.interpret_node(graph, graph.out_one(target, 'value'))
            field, _ = yield from self.interpret_node(graph, graph.out_one(target, 'slice'))
        elif isinstance(target, sst.GetAttr):
            target_upper, _ = yield from self.interpret_node(graph, graph.out_one(node, 'value'))
            field = node.name
        else:
            raise NotImplementedError()
        
        if node.op == '+':
            target_upper[field] += value
        elif node.op == '-':
            target_upper[field] -= value
        elif node.op == '*':
            target_upper[field] *= value
        elif node.op == '/':
            target_upper[field] /= value
        else:
            raise ValueError(node.op)

        return None, graph.out_one(node, 'next', optional=True)

    def _interpret_bin_op(self, graph, node):
        left, _ = yield from self.interpret_node(graph, graph.out_one(node, 'left'))
        right, _ = yield from self.interpret_node(graph, graph.out_one(node, 'right'))

        if node.op == 'in':
            left, right = right, left

        result = BIN_OP_MAP[node.op](
            left, right
        )

        return result, graph.out_one(node, 'next', optional=True)

    def _interpret_bool_op(self, graph, node):
        value_edges = graph.out_edges(node, 'values')
        value_edges.sort(key=lambda x: x.param)
        values = []
        for edge in value_edges:
            val, _ = yield from self.interpret_node(graph, edge.end)
            values.append(val)
        
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
        operand, _ = yield from self.interpret_node(graph, graph.out_one(node, 'operand'))

        if node.op == 'USub':
            result = -operand
        elif node.op == 'Not':
            result = not operand
        else:
            raise ValueError(node.op)

        return result, graph.out_one(node, 'next', optional=True)

    def interpret_node(self, graph: Graph, node):
        graph.active_node_id = node.node_id
        yield
        
        if isinstance(node, sst.FuncCall):
            val = yield from self._interpret_func_call(graph, node)
            return val

        elif isinstance(node, sst.If):
            val = yield from self._interpret_if(graph, node)
            return val

        elif isinstance(node, sst.While):
            val = yield from self._interpret_while(graph, node)
            return val

        elif isinstance(node, sst.For):
            val = yield from self._interpret_for(graph, node)
            return val

        elif isinstance(node, sst.Assign):
            val = yield from self._interpret_assign(graph, node)
            return val

        elif isinstance(node, sst.SelfAssign):
            val = yield from self._interpret_self_assign(graph, node)
            return val

        elif isinstance(node, sst.BinOp):
            val = yield from self._interpret_bin_op(graph, node)
            yield
            return val

        elif isinstance(node, sst.Compare):
            val = yield from self._interpret_bin_op(graph, node)
            return val

        elif isinstance(node, sst.BoolOp):
            val = yield from self._interpret_bool_op(graph, node)
            return val

        elif isinstance(node, sst.UnaryOp):
            val = yield from self._interpret_unary_op(graph, node)
            return val

        elif isinstance(node, sst.Constant):
            return node.value, graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.List):
            value = []
            elements = graph.out_edges(node, 'elements')
            for edge in elements:
                node_value, _ = yield from self.interpret_node(graph, edge.end)
                value.append(node_value)
            return value, graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.Tuple):
            value = []
            elements = graph.out_edges(node, 'elements')
            for edge in elements:
                node_value, _ = yield from self.interpret_node(graph, edge.end)
                value.append(node_value)
            return tuple(value), graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.Dict):
            result = {}
            keys = graph.out_edges(node, 'keys')
            values = graph.out_edges(node, 'values')
            for key, value in zip(keys, values):
                assert key.param == value.param
                
                node_key, _ = yield from self.interpret_node(graph, key.end)
                node_value, _ = yield from self.interpret_node(graph, value.end)

                result[node_key] = node_value

            return result, graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.Variable):
            return self.ctx[-1].get(node.name), graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.GetAttr):
            value, _ = yield from self.interpret_node(graph, graph.out_one(node, 'value'))
            return getattr(value, node.name), graph.out_one(node, 'next', optional=True)

        elif isinstance(node, sst.GetItem):
            value, _ = yield from self.interpret_node(graph, graph.out_one(node, 'value'))
            _slice, _ = yield from self.interpret_node(graph, graph.out_one(node, 'slice'))
            
            try:
                return value[_slice], graph.out_one(node, 'next', optional=True)
            except IndexError:
                breakpoint()
                pass

        elif isinstance(node, sst.Return):
            value_node = graph.out_one(node, 'value', optional=True)
            if value_node is None:
                self.ctx[-1].return_value = None
            else:
                self.ctx[-1].return_value, _ = yield from self.interpret_node(graph, value_node)
            
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

        elif isinstance(node, sst.Raise):
            exc, _ = yield from self.interpret_node(graph, graph.out_one(node, 'exc'))
            raise exc

        else:
            raise ValueError(node)

    def interpret_body(self, graph: Graph, head):
        assert head is not None
        result = None

        while head is not None:
            result, head = yield from self.interpret_node(graph, head)
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
        yield from self.interpret_body(graph, head)
        ctx = self.ctx.pop()

        if ctx.return_value is NO_RETURN_VALUE:
            return None
        return ctx.return_value

    def load_code(self, code):
        for ast_node in ast.parse(code).body:
            func_def: ast.FunctionDef = ast_node
            graph = ast_to_sst.Converter().convert(func_def)
            args = [
                (
                    arg.arg, 
                    _convert_ast_node_to_concept(arg.annotation),
                ) for arg in ast_node.args.args
            ]
            
            fdo = FunctionDispatchOption(
                args=args,
                graph=graph,
            )
            
            # args_hash = _args_hash_func(args)
            # func_id = f"{func_def.name}--{args_hash}"
            
            # graph_head = graph.get_nodes()[0]
            # self.combined_graph.merge(graph, graph_head, edge=None)
            # self.combined_graph.function_heads[func_id] = graph_head
                                    
            if func_def.name in self.global_vars:
                self.global_vars[func_def.name].dispatch_options.append(fdo)
            else:
                func = FunctionEntity(
                    interpreter=self,
                    name=func_def.name,
                    dispatch_options=[fdo]
                )
                self.global_vars[func_def.name] = func
                self.func_defs.append(func)
            
    def load_file(self, path: str):
        with open(path) as f:
            self.load_code(f.read())

    def run_function(self, name, args=None, kwargs=None):
        func = self.global_vars[name]
        graph, new_kwargs = func.resolve_dispatch(args or [], kwargs or {})
        return self.interpret_function(graph, graph.get_nodes()[0], new_kwargs)

    def run_main(self):
        return self.run_function('main')
    
    def dispatch_check_param_types(self, param_type, arg_value):
        return True, arg_value
