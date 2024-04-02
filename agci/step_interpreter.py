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
        
        self.func_defs = {}
        
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
            
            result = func(*params['args'], **params['kwargs'])
            
            _next = graph.out_one(node, 'next', optional=True)
            
            return _next, None, result
        
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
            value = self.global_vars[node.name]
            
        return None, None, value
    
    def _interpret_constant(self, graph: Graph, node: Constant, state):
        return None, None, node.value
    
    def _interpret_get_attr(self, graph: Graph, node: GetAttr, state):
        if state is None:
            value = graph.out_one(node, 'value')
            return value, {'value': value}, NO_VALUE
        else:
            return None, None, getattr(state['value'], node.name)
    
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
        
    def run(self):
        while True:
            try:
                self.step()
            except StopIteration:
                break
        
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
        resulting_kwargs = {}
        
        disp_opt = self.func_defs[name].dispatch_options[0]
        
        for arg_name, arg_concept in disp_opt.args:
            if args:
                if arg_name in kwargs:
                    raise ValueError(f'Argument "{arg_name}" is provided twice!')
                resulting_kwargs[arg_name] = args.pop(0)
            elif arg_name in kwargs:
                resulting_kwargs[arg_name] = kwargs.pop(arg_name)
                
        if kwargs:
            raise ValueError(f'Unknown arguments: {", ".join(kwargs.keys())}')
        
        if args:
            raise ValueError(f'Too many arguments: {", ".join(args)}')
        
        if len(resulting_kwargs) != len(disp_opt.args):
            raise ValueError('Not enough arguments!')
        
        frame = ExecutionFrame(
            head=disp_opt.graph.get_nodes()[0],
            graph=disp_opt.graph,
            context=[InterpreterContext(variables=resulting_kwargs)],
        )
        self.frame_stack.append(frame)

    def dispatch_check_param_types(self, param_type, arg_value):
        return True, arg_value
    
    def add_context(self, variables):
        self.frame_stack[-1].context.append(InterpreterContext(variables=variables))