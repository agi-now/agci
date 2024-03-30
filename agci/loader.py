import ast
from agci.sst import ast_to_sst
from agci.sst.entities import (
    FunctionEntity,
    FunctionDispatchOption, 
)


def load_code(code: str) -> list[FunctionEntity]:
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


def load_file(path: str) -> list[FunctionEntity]:
    with open(path) as f:
        return load_code(f.read())
