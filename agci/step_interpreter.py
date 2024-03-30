import ast
from dataclasses import dataclass
from typing import Optional

from agci.interpreter import (
    BIN_OP_MAP, 
    NO_RETURN_VALUE, 
    InterpreterContext,
)
from agci.sst.entities import (
    Constant,
    FuncCall, 
    FunctionDispatchOption,
    Node,
    Variable,
)

from . import sst
from .sst import Graph, FunctionEntity
from .sst import ast_to_sst


NO_VALUE = object()


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


class StepInterpreter:
    def __init__(self, global_vars: dict[str, any]):
        global_vars['__agci'] = self
        self.global_vars = global_vars
        self.ctx = []
        self._head = None
        self._head_graph = None
        self._back_node_stack = []
        self._params_stack = []
        self._value = None
        
        self.func_defs = {}
        
    def step(self):
        params = None
        key1, key2, head = self._get_head_from_params()
        
        if head is None:
            head = self._head
            
        if head is None and self._params_stack:
            head, params = self._params_stack.pop()
            
        if head is None and self._back_node_stack:
            head, params = self._back_node_stack.pop()
            
        if head is None:
            print('exit')
            breakpoint()
            pass
        
        self._head, new_params, result = self._interpret_node(self._head_graph, head, params)
        
        if result is not NO_VALUE and self._params_stack:
            if key2 is not None:
                self._params_stack[-1][1][key1][key2] = result
            else:
                self._params_stack[-1][1][key1] = result
        
        if new_params is not None:
            self._params_stack.append((head, new_params))
        
    def _get_head_from_params(self):
        if not self._params_stack:
            return None, None, None
        for key, value in self._params_stack[-1][1].items():
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
        
    def _interpret_node(self, graph: Graph, node: Node, params: Optional[dict]):
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
            
            params = {
                'func': func_node,
                'args': args_nodes,
                'kwargs': kwargs_nodes,
            }
            
            return None, params, NO_VALUE
        else:
            func = params['func']
            result = func(*params['args'], **params['kwargs'])
            
            _next = graph.out_one(node, 'next', optional=True)
            
            return _next, None, result

    def _interpret_variable(self, graph: Graph, node: Variable, state):
        return None, None, self.global_vars[node.name]
    
    def _interpret_constant(self, graph: Graph, node: Constant, state):
        return None, None, node.value
    
    def _interpret_if(self, graph: Graph, node: sst.If, state):
        if state is None:
            test = graph.out_one(node, 'test')
            return None, {
                'test': test,
                'done': False,               
            }, NO_VALUE
        elif not state['done']:
            state['done'] = True
            self._back_node_stack.append((node, state))
            if state['test']:
                return graph.out_one(node, 'next', True), None, NO_VALUE
            else:
                return graph.out_one(node, 'next', False), None, NO_VALUE
        else:
            return graph.out_one(node, 'next', optional=True), None, NO_VALUE
    
    def _interpret_assign(self, graph: Graph, node: sst.Assign, state):
        if state is None:
            target = graph.out_many(node, 'targets')[0]
            value = graph.out_one(node, 'value')
            
            if isinstance(target, Variable):
                target, attr = None, target.name

            return None, {
                'target': target,
                'attr': attr,
                'value': value,
            }, NO_VALUE
        else:
            target = state['target']
            attr = state['attr']
            
            if target is None:
                self.global_vars[attr] = state['value']
            
            return graph.out_one(node, 'next', optional=True), None, NO_VALUE
    
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
        
    def trigger_function(self, name, *args, **kwargs):
        self._back_node_stack.clear()
        self._params_stack.clear()
        func_def = self.func_defs[name]
        self._head_graph = func_def.dispatch_options[0].graph
        self._head = self._head_graph.get_nodes()[0]

    def dispatch_check_param_types(self, param_type, arg_value):
        return True, arg_value
