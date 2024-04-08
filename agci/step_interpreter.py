import ast
import operator
from typing import Optional
from dataclasses import dataclass, field

from agci.sst.entities import (
    Node,
    Constant,
    FuncCall,
    Variable,
    GetAttr,
    FunctionDispatchOption,
)

from . import sst
from .sst import Graph, FunctionEntity, uuid
from .sst import ast_to_sst


NO_VALUE = object()

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
    '%': operator.mod,
}


@dataclass
class InterpreterContext:
    variables: dict[str, any]
    return_value = NO_VALUE

    def get(self, name):
        try:
            return self.variables[name]
        except KeyError as e:
            raise KeyError(f'Variable "{name}" not found!') from e


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


def camel_to_snake_case(name):
    return ''.join([
        f'_{c.lower()}' if c.isupper() else c
        for c in name
    ])[1:]
    
    
@dataclass(slots=True)
class NodeState:
    node: Node
    data: dict[str, any]
    
    
@dataclass(slots=True)
class ExecutionFrame:
    head: Node
    graph: Graph
    back_node_stack: list[tuple[Node, dict[str, any]]] = field(default_factory=list)
    params_stack: list[tuple[Node, dict[str, any]]] = field(default_factory=list)
    value: any = None
    context: list[InterpreterContext] = field(default_factory=list)
    return_value: any = NO_VALUE
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def get_head_from_params(self):
        if not self.params_stack:
            return None, None, None
        for key, value in self.params_stack[-1][1].items():
            if isinstance(value, Node):
                return key, None, value
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, Node):
                        return key, idx, item
            elif isinstance(value, dict):
                for dct_key, item in value.items():
                    if isinstance(item, Node):
                        return key, dct_key, item
        return None, None, None
    
    def set_result(self, key1, key2, result):
        if result is not NO_VALUE and self.params_stack:
            if key2 is not None:
                self.params_stack[-1][1][key1][key2] = result
            else:
                self.params_stack[-1][1][key1] = result


class StepInterpreter:
    def __init__(self, global_vars: dict[str, any]):
        global_vars['__agci'] = self
        self.global_vars = global_vars
        self.frame_stack = []
        
        self.func_defs: dict[str, FunctionEntity] = {}
        
    def step(self):
        if not self.frame_stack:
            raise StopIteration
        
        frame = self.frame_stack[-1]
        
        params = None
        key1, key2, head = frame.get_head_from_params()
        
        if head is None:
            head = frame.head
        
        if head is None and frame.params_stack:
            head, params, key1, key2 = frame.params_stack.pop()
        
        if head is None and frame.back_node_stack:
            head, params = frame.back_node_stack.pop()
        
        if head is None:
            if self.frame_stack:
                frame = self.frame_stack.pop()
                return_value = frame.return_value
                if return_value is NO_VALUE:
                    return_value = None
                if self.frame_stack and self.frame_stack[-1].params_stack:
                    if '__frames' in self.frame_stack[-1].params_stack[-1][-3]:
                        self.frame_stack[-1].params_stack[-1][-3]['__frames'][frame.id] = return_value
                return
            else:
                raise StopIteration
        
        frame.head, new_params, result = self._interpret_node(frame.graph, head, params)
        
        frame.set_result(key1, key2, result)
        
        if new_params is not None:
            frame.params_stack.append((head, new_params, key1, key2))
        
    def _interpret_node(self, graph: Graph, node: Node, params: Optional[dict]):
        """
        Returns:
            - next node
            - params (state) of the node
            - result of the interpretation
            
        If params is not None, the node will be called again with the params evaluated.
        """
        node_type = camel_to_snake_case(node.__class__.__name__)
        node_handler_name = f'_interpret_{node_type}'
        try:
            node_handler = getattr(self, node_handler_name)
        except AttributeError:
            raise NotImplementedError(node_handler_name)
        
        return node_handler(graph, node, params)
    
    def _interpret_func_call(self, graph: Graph, node: FuncCall, params: Optional[dict]):
        if params is None:
            func_node = graph.out_one(node, 'func')
            args_nodes_dict = graph.out_dict(node, 'args')
            args_nodes = [args_nodes_dict[idx] for idx in sorted(args_nodes_dict.keys())]
            kwargs_nodes = graph.out_dict(node, 'kwargs')
            
            if func_node.name == 'range':
                breakpoint()
            
            params = {
                'func': func_node,
                'args': args_nodes,
                'kwargs': kwargs_nodes,
            }
            
            return None, params, NO_VALUE
        if "__frames" in params:
            return None, None, params['__frames'][params['result_frame_id']]
        else:
            func = params['func']
            if isinstance(func, FunctionEntity):
                graph, new_kwargs = func.resolve_dispatch(params['args'], params['kwargs'])
                frame = ExecutionFrame(
                    head=graph.get_nodes()[0],
                    graph=graph,
                    context=[InterpreterContext(variables=new_kwargs)],
                )
                self.frame_stack.append(frame)
                return None, {
                    '__frames': {
                        frame.id: frame,
                    },
                    'result_frame_id': frame.id,
                }, NO_VALUE
            
            if func is range:
                breakpoint()
            result = func(*params['args'], **params['kwargs'])
            
            _next = graph.out_one(node, 'next', optional=True)
            
            return _next, None, result
        
    def _interpret_dict(self, graph: Graph, node: sst.Dict, state):
        if state is None:
            keys = [
                edge.end for edge in sorted(
                    graph.out_edges(node, 'keys'),
                    key=lambda x: x.param
                )
            ]
            values = [
                edge.end for edge in sorted(
                    graph.out_edges(node, 'values'),
                    key=lambda x: x.param
                )
            ]
            return None, {
                "keys": keys,
                "values": values,
            }, NO_VALUE
        else:
            _next = graph.out_one(node, 'next', optional=True)
            keys = state['keys']
            values = state['values']
            return _next, None, {
                key: value
                for key, value in zip(keys, values)
            }
            
    def _interpret_list(self, graph: Graph, node: sst.List, state):
        if state is None:
            values = [
                edge.end for edge in sorted(
                    graph.out_edges(node, 'values'),
                    key=lambda x: x.param
                )
            ]
            return None, {
                "values": values,
            }, NO_VALUE
        else:
            _next = graph.out_one(node, 'next', optional=True)
            return _next, None, state['values']
        
    def _interpret_return(self, graph: Graph, node: sst.Return, state):
        if state is None:
            value = graph.out_one(node, 'value')
            return None, {'value': value}, NO_VALUE
        else:
            self.frame_stack[-1].return_value = state['value']
            return None, None, NO_VALUE
        
    def _interpret_variable(self, graph: Graph, node: Variable, state):
        try:
            value = self.frame_stack[-1].context[-1].variables[node.name]
        except (KeyError, IndexError):
            try:
                value = self.global_vars[node.name]
            except KeyError:
                breakpoint()
                pass
            
        return None, None, value
    
    def _interpret_constant(self, graph: Graph, node: Constant, state):
        return None, None, node.value
    
    def _interpret_get_attr(self, graph: Graph, node: GetAttr, state):
        if state is None:
            value = graph.out_one(node, 'value')
            return value, {'value': value}, NO_VALUE
        else:
            return None, None, getattr(state['value'], node.name)
    
    def _interpret_get_item(self, graph: Graph, node: sst.GetItem, state):
        if state is None:
            value = graph.out_one(node, 'value')
            key = graph.out_one(node, 'slice')
            return None, {
                'value': value,
                'slice': key,
            }, NO_VALUE
        else:
            return None, None, state['value'][state['slice']]
    
    def _interpret_if(self, graph: Graph, node: sst.If, state):
        if state is None:
            test = graph.out_one(node, 'test')
            return None, {
                'test': test,
                'done': False,               
            }, NO_VALUE
        elif not state['done']:
            state['done'] = True
            self.frame_stack[-1].back_node_stack.append((node, state))
            if state['test']:
                return graph.out_one(node, 'next', True), None, NO_VALUE
            else:
                return graph.out_one(node, 'next', False, optional=True), None, NO_VALUE
        else:
            return graph.out_one(node, 'next', optional=True), None, NO_VALUE
        
    def _interpret_for(self, graph: Graph, node: sst.If, state):
        if state is None or state['iter'] is None:
            test = graph.out_one(node, 'iter')
            return None, {
                'iter': test,
            }, NO_VALUE
        else:
            # make iter an iterator if it's not
            if not hasattr(state['iter'], '__next__'):
                state['iter'] = iter(state['iter'])
                        
            try:
                value = next(state['iter'])
                self.frame_stack[-1].context[-1].variables[node.target] = value
                self.frame_stack[-1].back_node_stack.append((node, state))
            except StopIteration:
                return graph.out_one(node, 'next', optional=True), None, NO_VALUE
                
            return graph.out_one(node, 'next', True), None, NO_VALUE
    
    def _interpret_end_for(self, graph: Graph, node: sst.EndFor, state):
        return graph.out_one(node, 'next', optional=True), None, NO_VALUE
    
    def _interpret_while(self, graph: Graph, node: sst.While, state):
        if state is None or state['test'] is None:
            test = graph.out_one(node, 'test')
            return None, {
                'test': test,
            }, NO_VALUE
        else:
            if state['test']:
                state['test'] = None
                self.frame_stack[-1].back_node_stack.append((node, state))
                return graph.out_one(node, 'next', True), None, NO_VALUE
            else:
                return graph.out_one(node, 'next', optional=True), None, NO_VALUE
    
    def _interpret_end_while(self, graph: Graph, node: sst.EndWhile, state):
        return graph.out_one(node, 'next', optional=True), None, NO_VALUE
    
    def _interpret_compare(self, graph: Graph, node: sst.Compare, state):
        if state is None:
            left = graph.out_one(node, 'left')
            right = graph.out_one(node, 'right')
            op = node.op
            return None, {
                'left': left,
                'right': right,
                'op': op,
            }, NO_VALUE
        else:
            return None, None, BIN_OP_MAP[state['op']](state['left'], state['right'])
    
    def _interpret_bin_op(self, graph: Graph, node: sst.BinOp, state):
        if state is None:
            left = graph.out_one(node, 'left')
            right = graph.out_one(node, 'right')
            op = node.op
            
            return None, {
                'left': left,
                'right': right,
                'op': op,
            }, NO_VALUE
        else:
            return None, None, BIN_OP_MAP[state['op']](state['left'], state['right'])
    
    def _interpret_assign(self, graph: Graph, node: sst.Assign, state):
        if state is None:
            target = graph.out_many(node, 'targets')[0]
            value = graph.out_one(node, 'value')
            
            if isinstance(target, Variable):
                target, attr = None, target.name
                _type = 'var'
            elif isinstance(target, GetAttr):
                target = graph.out_one(target, 'value')
                attr = target.name
                _type = 'attr'
            elif isinstance(target, sst.GetItem):
                attr = graph.out_one(target, 'slice')
                target = graph.out_one(target, 'value')
                _type = 'item'
            else:
                raise NotImplementedError(target)

            return None, {
                'target': target,
                'attr': attr,
                'value': value,
                'type': _type,
            }, NO_VALUE
        else:
            target = state['target']
            attr = state['attr']
            
            match state['type']:
                case 'var':
                    self.global_vars[attr] = state['value']
                case 'attr':
                    setattr(target, attr, state['value'])
                case 'item':
                    target[attr] = state['value']
            
            return graph.out_one(node, 'next', optional=True), None, NO_VALUE
        
    def run(self):
        idx = 0
        while True:
            idx += 1
            try:
                self.step()
            except StopIteration:
                break
        print(f'Finished in {idx} steps')
        
    def load_file(self, path: str):
        with open(path) as f:
            self.load_code(f.read())
    
    def load_code(self, code: str):
        for func_def in ast.parse(code).body:
            graph = ast_to_sst.convert(func_def)
            dispatch_option = FunctionDispatchOption(
                args=[
                    (
                        arg.arg, 
                        _convert_ast_node_to_concept(arg.annotation),
                    ) for arg in func_def.args.args
                ],
                graph=graph,   
            )
            if func_def.name in self.func_defs:
                self.func_defs[func_def.name].dispatch_options.append(dispatch_option)
            else:
                self.func_defs[func_def.name] = FunctionEntity(
                    name=func_def.name,
                    dispatch_options=[dispatch_option],
                )
            self.global_vars[func_def.name] = self.func_defs[func_def.name]
            self.func_defs[func_def.name].interpreter = self
        
    def trigger_function(self, name, *args, **kwargs):
        graph, new_kwargs = self.func_defs[name].resolve_dispatch(args, kwargs)
        
        frame = ExecutionFrame(
            head=graph.get_nodes()[0],
            graph=graph,
            context=[InterpreterContext(variables=new_kwargs)],
        )
        self.frame_stack.append(frame)
        
    def run_main(self):
        self.trigger_function('main')
        self.run()

    def dispatch_check_param_types(self, param_type, arg_value):
        raise NotImplementedError(param_type, arg_value)
    
    def add_context(self, variables):
        self.frame_stack[-1].context.append(InterpreterContext(variables=variables))