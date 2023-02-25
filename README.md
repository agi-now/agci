# Agent Code Interpreter

Library to convert python code to traversible [AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree) and back to code.  
By traversible AST we mean AST where every node has "next" edge pointing to the next node to execute.
Branching nodes have separate "next" edges for True and False condition.
Regular python AST is not actually represented as graph, so we need to do this converions:
Python -> Python AST -> Traversible AST.
Traversible AST is then being interpreted by the interepter.  
We need AST to be traversible in order to do pattern matching on it, traversability is not a requirement for running the code.
