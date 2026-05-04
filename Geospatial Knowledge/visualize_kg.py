from pyvis.network import Network
import networkx as nx
import json

# ==============================
# Load Knowledge Graph
# ==============================
with open("outputs/geospatial_knowledge_graph_final.json") as f:
    graph_data = json.load(f)

G = nx.node_link_graph(graph_data)

print(f"Loaded graph with {G.number_of_nodes()} nodes")


# ==============================
# Select ONLY 10 Cities
# ==============================
cities = [
    n for n, d in G.nodes(data=True)
    if d.get("type") == "City"
]  # ⭐ change number here

sub_nodes = set(cities)

# include neighbors of cities
for city in cities:
    sub_nodes.update(G.neighbors(city))

subG = G.subgraph(sub_nodes).copy()

print(f"Subgraph nodes: {subG.number_of_nodes()}")


# ==============================
# Create Interactive Network
# ==============================
net = Network(
    height="800px",
    width="100%",
    directed=True,
    notebook=False
)

net.toggle_physics(True)


# ==============================
# Add Nodes FIRST
# ==============================
added_nodes = set()

for node, data in subG.nodes(data=True):

    node_id = str(node)

    title = f"""
    <b>Name:</b> {node_id}<br>
    <b>Type:</b> {data.get('type')}<br>
    <b>Lat:</b> {data.get('lat')}<br>
    <b>Lon:</b> {data.get('lon')}
    """

    net.add_node(
        node_id,
        label=node_id,
        title=title
    )

    added_nodes.add(node_id)


# ==============================
# Add Edges SAFELY
# ==============================
for u, v, data in subG.edges(data=True):

    u_id = str(u)
    v_id = str(v)

    # ⭐ CRITICAL FIX
    if u_id in added_nodes and v_id in added_nodes:
        net.add_edge(
            u_id,
            v_id,
            title=data.get("relation", "")
        )


# ==============================
# Better Physics Layout
# ==============================
net.force_atlas_2based(
    gravity=-50,
    central_gravity=0.01,
    spring_length=150,
    spring_strength=0.08,
    damping=0.4
)


# ==============================
# Enable UI Controls (VERY NICE)
# ==============================
net.show_buttons(filter_=["physics"])


# ==============================
# Save Interactive HTML
# ==============================
output_file = "interactive_kg.html"
net.write_html(output_file)

print(f"\n✅ Interactive graph saved: {output_file}")
print("👉 Open it in Chrome and DRAG nodes freely!")