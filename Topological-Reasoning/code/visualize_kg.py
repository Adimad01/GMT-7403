import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def load_kg(file_path):
    """Charge le fichier JSON du Knowledge Graph."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_geographic_context(kg_data, place_a=None, place_b=None, save_path="kg_visualization.png", max_nodes=10):
    """
    Visualise le graphe. 
    Si place_a et place_b sont fournis, visualise uniquement le sous-graphe pertinent.
    """
    G = nx.MultiDiGraph()
    
    # Configuration des couleurs par type
    color_map = {
        "city": "#3b82f6",      # Bleu
        "county": "#ef4444",    # Rouge
        "water": "#06b6d4",     # Cyan
        "park": "#10b981",      # Vert
        "place": "#8b5cf6",     # Violet (par défaut)
        "geometry_type": "#94a3b8" # Gris (Ontologie)
    }

    # Filtrage des données si demandé
    relevant_nodes = set()
    if place_a and place_b:
        # Always include the two focus places
        relevant_nodes.add(place_a)
        relevant_nodes.add(place_b)
        # Collect immediate neighbors
        for link in kg_data['links']:
            if link['source'] == place_a or link['target'] == place_a:
                relevant_nodes.add(link['source'])
                relevant_nodes.add(link['target'])
            if link['source'] == place_b or link['target'] == place_b:
                relevant_nodes.add(link['source'])
                relevant_nodes.add(link['target'])
        # If too many nodes, trim while keeping place_a/place_b and their neighbors first
        if len(relevant_nodes) > max_nodes:
            preserved = {place_a, place_b}
            # keep immediate neighbors
            neighbors = [n for n in relevant_nodes if n not in preserved]
            neighbors_sorted = sorted(neighbors)[: max_nodes - len(preserved)]
            relevant_nodes = preserved.union(neighbors_sorted)
    else:
        # Otherwise sample deterministically up to max_nodes for readability
        relevant_nodes = {node['id'] for node in kg_data['nodes'][:max_nodes]}

    # Ajout des nœuds
    node_colors = []
    node_labels = {}
    
    for node in kg_data['nodes']:
        if node['id'] in relevant_nodes:
            category = node.get('category', 'place')
            placetype = node.get('placetype', category)
            
            G.add_node(node['id'], type=placetype)
            node_colors.append(color_map.get(placetype, color_map["place"]))
            node_labels[node['id']] = node['id'].split(',')[0] # Label court

    # Ajout des liens
    edge_labels = {}
    for link in kg_data['links']:
        if link['source'] in relevant_nodes and link['target'] in relevant_nodes:
            # On distingue les liens spatiaux (DE-9IM) des liens ontologiques (Geometry)
            label = link.get('spatial_logic') or link.get('predicate') or ""
            G.add_edge(link['source'], link['target'], label=label)
            edge_labels[(link['source'], link['target'])] = label

    # Dessin du graphe
    plt.figure(figsize=(14, 10))
    # For small graphs a Kamada-Kawai layout often looks better; fall back to spring otherwise
    try:
        if len(G.nodes) <= max_nodes:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    except Exception:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Noeuds
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color=node_colors, alpha=0.9)
    
    # Labels des noeuds
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
    
    # Arcs
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='#cbd5e1', arrowsize=20)
    
    # Labels des arcs
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='#475569')

    title = f"Visualisation du KG Géospatial"
    if place_a and place_b:
        title += f" : {place_a} ↔ {place_b}"
    
    plt.title(title, pad=20, fontsize=15)
    plt.axis('off')
    
    # Légende manuelle
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=k,
                              markerfacecolor=v, markersize=10) for k, v in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper left', title="Légende Types")

    print(f"Saving visualization to {save_path}...")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def visualize_geographic_context_plotly(kg_data, place_a=None, place_b=None, save_path="kg_visualization.html", max_nodes=10):
    """
    Create an interactive Plotly HTML visualization of the KG.
    Saves to `save_path` and also calls `fig.show()` when possible.
    """
    import plotly.graph_objects as go

    G = nx.MultiDiGraph()

    # Same color map as Matplotlib version
    color_map = {
        "city": "#3b82f6",
        "county": "#ef4444",
        "water": "#06b6d4",
        "park": "#10b981",
        "place": "#8b5cf6",
        "geometry_type": "#94a3b8"
    }

    # Build relevant nodes set (reuse logic from matplotlib function)
    relevant_nodes = set()
    if place_a and place_b:
        relevant_nodes.add(place_a)
        relevant_nodes.add(place_b)
        for link in kg_data['links']:
            if link['source'] == place_a or link['target'] == place_a:
                relevant_nodes.add(link['source'])
                relevant_nodes.add(link['target'])
            if link['source'] == place_b or link['target'] == place_b:
                relevant_nodes.add(link['source'])
                relevant_nodes.add(link['target'])
        # Trim to max_nodes keeping focus nodes
        if len(relevant_nodes) > max_nodes:
            preserved = {place_a, place_b}
            neighbors = [n for n in relevant_nodes if n not in preserved]
            neighbors_sorted = sorted(neighbors)[: max_nodes - len(preserved)]
            relevant_nodes = preserved.union(neighbors_sorted)
    else:
        relevant_nodes = {node['id'] for node in kg_data['nodes'][:max_nodes]}

    node_labels = {}
    node_colors = []
    for node in kg_data['nodes']:
        if node['id'] in relevant_nodes:
            category = node.get('category', 'place')
            placetype = node.get('placetype', category)
            G.add_node(node['id'], type=placetype)
            node_colors.append(color_map.get(placetype, color_map['place']))
            node_labels[node['id']] = node['id'].split(',')[0]

    edge_texts = {}
    for link in kg_data['links']:
        if link['source'] in relevant_nodes and link['target'] in relevant_nodes:
            label = link.get('spatial_logic') or link.get('predicate') or ""
            G.add_edge(link['source'], link['target'], label=label)
            edge_texts[(link['source'], link['target'])] = label

    # Layout positions - prefer Kamada-Kawai for small graphs
    try:
        if len(G.nodes) <= max_nodes:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    except Exception:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(data.get('label', ''))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#cbd5e1'),
        hoverinfo='none',
        mode='lines'
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_color_list = []
    for i, n in enumerate(G.nodes()):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_labels.get(n, n))
        # Map color by node attribute
        node_color_list.append(color_map.get(G.nodes[n].get('type', 'place'), color_map['place']))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='top center',
        marker=dict(showscale=False, color=node_color_list, size=20, line_width=1)
    )

    # Optionally add hover annotations for edges by placing invisible scatter points at midpoints
    edge_mid_x = []
    edge_mid_y = []
    edge_mid_text = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_mid_x.append((x0 + x1) / 2)
        edge_mid_y.append((y0 + y1) / 2)
        edge_mid_text.append(data.get('label', ''))

    edge_mid_trace = go.Scatter(
        x=edge_mid_x,
        y=edge_mid_y,
        mode='markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        hoverinfo='text',
        text=edge_mid_text,
    )

    fig = go.Figure(data=[edge_trace, edge_mid_trace, node_trace],
                    layout=go.Layout(
                        title=(f"Visualisation du KG Géospatial: {place_a} ↔ {place_b}" if place_a and place_b else "Visualisation du KG Géospatial"),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    print(f"Saving interactive visualization to {save_path}...")
    fig.write_html(save_path, include_plotlyjs='cdn')
    try:
        fig.show()
    except Exception:
        # In headless environments fig.show may fail; file is still written
        pass

if __name__ == "__main__":
    # Remplacez par le chemin de votre fichier
    FILE_PATH = 'results/knowledge_graph.json'
    
    try:
        data = load_kg(FILE_PATH)
        # Exemple 1 : Visualisation globale (échantillon)
        # Prefer Plotly (interactive) if available, otherwise fallback to Matplotlib
        try:
            import plotly
            visualize_geographic_context_plotly(data, save_path="kg_global.html")
        except Exception:
            # If Plotly isn't installed or fails, fallback to static Matplotlib PNG
            visualize_geographic_context(data, save_path="kg_global.png")
        
        # Exemple 2 : Focus sur une paire spécifique de votre CSV
        # To generate an interactive focus view, uncomment the next line and install plotly
        # visualize_geographic_context_plotly(data, place_a="Edna, Texas", place_b="Lake Texana, Texas", save_path="kg_focus.html")
        
    except FileNotFoundError:
        print(f"Erreur : Le fichier {FILE_PATH} est introuvable.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")