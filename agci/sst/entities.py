import uuid

from typing import Optional, Self
from dataclasses import dataclass, field


@dataclass
class FunctionDispatchOption:
    args: list[tuple[str, Optional[str]]]
    graph: 'Graph'
    
    @property
    def dispatch_id(self):
        breakpoint()
        pass
    

@dataclass
class FunctionEntity:
    name: str
    dispatch_options: list[FunctionDispatchOption]
    interpreter: 'Interpreter' = None
    
    def _collect_args_no_dispatch(self, args, kwargs):
        opt = self.dispatch_options[0]
        args_copy = list(args).copy()
        result = {}
        
        for param_name, _ in opt.args:
            if param_name in kwargs:
                arg_value = kwargs[param_name]
            else:
                arg_value = args_copy.pop(0)
                
            result[param_name] = arg_value
            
        return result
            
    def resolve_dispatch(self, args, kwargs):
        for opt in self.dispatch_options:
            for _, param_type in opt.args:
                if param_type is None:
                    assert len(self.dispatch_options) == 1, "Dispatch requires param types"
                    resulting_kwargs = self._collect_args_no_dispatch(args, kwargs)
                    return opt.graph, resulting_kwargs
        
        for opt in self.dispatch_options:
            resulting_kwargs = {}
            args_copy = list(args).copy()
            matches = True
            
            for param_name, param_type in opt.args:
                if param_name in kwargs:
                    arg_value = kwargs[param_name]
                else:
                    if not args_copy:
                        raise ValueError(f"No arguments left: {self.name}")
                    arg_value = args_copy.pop(0)
                if param_type is None:
                    continue
                
                arg_ok, new_arg_value = self.interpreter.dispatch_check_param_types(
                    param_type,
                    arg_value,
                )
                
                if not arg_ok:
                    matches = False
                    break
                
                resulting_kwargs[param_name] = new_arg_value
                
            if matches:
                return opt.graph, resulting_kwargs
        raise ValueError(f"No matching dispatch option found for '{self.name}'")

    def __call__(self, *args, **kwargs):
        graph, new_kwargs = self.resolve_dispatch(args, kwargs)
        return self.interpreter.interpret_function(graph, graph.get_nodes()[0], new_kwargs)


class Node:
    node_id: str


@dataclass(slots=True, frozen=True)
class Edge:
    start: 'Node'
    end: 'Node'
    label: str
    param: any = None
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


class Graph:
    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self._nodes: list[Node] = []
        self._edges: list[Edge] = []
        self._nodes_by_id = {}
        self._edges_by_id = {}
        self.active_node_id = nodes[0].node_id if nodes else None
        self._unclosed_graph_heads = []
        self.function_heads = {}
        self.node_to_function_head = {}

        for node in nodes:
            self.add_node(node)

        for edge in edges:
            self.add_edge(edge)

    def find_node(self, node_id) -> Node:
        return self._nodes_by_id[node_id]

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def add_node(self, node: Node):
        self._nodes.append(node)
        self._nodes_by_id[node.node_id] = node

    def add_edge(self, edge: Edge):
        self._edges.append(edge)
        self._edges_by_id[edge.edge_id] = edge

    def in_(self, node, edge_label, param=None):
        results = []
        for edge in self._edges:
            if edge.end is node and edge.label == edge_label:
                if param is not None and edge.param is None:
                    continue
                if param is None and edge.param is not None:
                    continue
                if param != edge.param:
                    continue
                results.append(edge.start)

        return results
    
    def out_many(self, node, edge_label, param=None):
        results = []
        for edge in self._edges:
            if edge.start is node and edge.label == edge_label:
                if param is not None and edge.param is None:
                    continue
                if param is None and edge.param is not None:
                    continue
                if param != edge.param:
                    continue
                results.append(edge.end)

        return results

    def out_one(self, node, edge_label, param=None, optional=False):
        results = self.out_many(node, edge_label, param)
        if len(results) != 1:
            if optional:
                return None
            raise ValueError()

        return results[0]

    def out_edges(self, node, edge_label) -> list[Edge]:
        results = []
        for edge in self._edges:
            if edge.start is node and edge.label == edge_label:
                results.append(edge)

        return results
    
    def out_dict(self, node, edge_label):
        results = {}
        for edge in self._edges:
            if edge.start is node and edge.label == edge_label:
                results[edge.param] = edge.end

        return results
    
    def _find_where_to_merge(self, head: Node):
        if self.out_one(head, 'next', True, optional=True) is None:
            return head
        breakpoint()
        pass
    
    def merge(self, graph: Self, head: Node, edge="next"):
        graph_head = graph.get_nodes()[0]
        
        for node in graph.get_nodes():
            self.add_node(node)
            
        for edge_ in graph.get_edges():
            self.add_edge(edge_)

        if edge is not None:
            self._edges.append(Edge(head, graph_head, edge))
        else:
            for node in graph.get_nodes():
                self.node_to_function_head[node.node_id] = head.node_id
        

@dataclass(slots=True, frozen=True)
class Assign(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class For(Node):
    target: str
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class EndFor(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class ElseFor(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class While(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class EndWhile(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class ElseWhile(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Dict(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class List(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Tuple(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Set(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class FuncCall(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class GetAttr(Node):
    name: str
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Variable(Node):
    name: str
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class GetItem(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class BinOp(Node):
    op: str
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class UnaryOp(Node):
    op: str
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Compare(Node):
    op: str
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Constant(Node):
    value: any
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class If(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Return(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Break(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)
    
    
@dataclass(slots=True, frozen=True)
class Continue(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class BoolOp(Node):
    op: str

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class SelfAssign(Node):
    op: str
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)


@dataclass(slots=True, frozen=True)
class Raise(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)
    
    
@dataclass(slots=True, frozen=True)
class Pass(Node):
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)