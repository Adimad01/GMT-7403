import pandas as pd
import json
import networkx as nx

# 1. Load Data
full_df = pd.read_pickle('worldcities100k.pkl') 
with open('outputs/subset_ids_gptoss_30pct.json', 'r') as f: 
    holdout_ids = set(json.load(f))

kg_df = full_df[~full_df.index.isin(holdout_ids)].copy()

# 2. Initialize Directed Graph
G = nx.DiGraph()

print("Building Knowledge Graph with Unique City Identities...")

for _, row in kg_df.iterrows():
    # Variables for clarity
    raw_name = row['AccentCity']
    country_code = row['Country'].upper()
    
    # CREATE UNIQUE ID: Combine name and country (e.g., "Paris_FR")
    # This prevents different cities with the same name from merging.
    city_unique_id = f"{raw_name}_{country_code}"
    
    # 3. Add City Node
    G.add_node(city_unique_id, 
               type='City', 
               display_name=raw_name, # The pretty name for the LLM
               lat=row['Latitude'], 
               lon=row['Longitude'], 
               pop=row['Population'])
    
    # 4. Add Country Node
    G.add_node(country_code, type='Country')
    
    # 5. Connect them: City_ID -> Country
    G.add_edge(city_unique_id, country_code, relation='LOCATED_IN')

print(f"Graph built with {G.number_of_nodes()} unique nodes.")

# 6. Verification: Search for "Paris"
paris_nodes = [n for n in G.nodes if n.startswith("Paris_")]
print(f"Found unique Paris entries: {paris_nodes}")

# 7. Export
output_path = 'outputs/geospatial_knowledge_graph_final.json'
graph_data = nx.node_link_data(G)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, indent=4, ensure_ascii=False)

print("✅ Fixed! Duplicate city names are now treated as separate entities.")