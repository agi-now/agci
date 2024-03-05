import networkx as nx
from agci.sst.entities import Edge, Node
from pyvis.network import Network


def to_html(nodes: list[Node], edges: list[Edge], head_id=None):
    node_by_id = {
        node.node_id: node
        for node in nodes
    }
    edge_by_id = {
        edge.edge_id: edge
        for edge in edges
    }
    G = nx.DiGraph()

    for edge in edges:
        G.add_node(edge.start.node_id)
        G.add_node(edge.end.node_id)
        G.add_edge(edge.start.node_id, edge.end.node_id, label=edge.edge_id)
        
    # Create a PyVis network
    net = Network(directed=True, notebook=True, height="100vh", width="100vw", bgcolor="#222222", font_color="white", cdn_resources='in_line')
    net.from_nx(G)

    for node in net.nodes:
        graph_node = node_by_id[node['id']]
        node["label"] = f"{graph_node}"
        node["title"] = f"{graph_node}"
        node['size'] = 30 if graph_node.node_id == head_id else 15
        
    for edge in net.edges:
        graph_edge = edge_by_id[edge['label']]
        edge["label"] = graph_edge.label + (f"[{graph_edge.param}]" if graph_edge.param is not None else "")
        edge['width'] = 5 if graph_edge.label == "next" else 1

    net.set_options("""
    var options = {
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "interaction": {
        "hover": true
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.4,
          "springLength": 50,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 1
        }
      }
    }
    """)

    net.show("graph.html")
