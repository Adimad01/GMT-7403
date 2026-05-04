import pandas as pd
import json

# 1. Chargement des données
df = pd.read_csv('../dataset/triplet_update_v3_70.csv')

nodes = {}
links = []

for _, row in df.iterrows():
    subj_name = str(row['place_name_subject']).strip()
    obj_name = str(row['place_name_object']).strip()
    subj_type = str(row['placetype_subject']).strip()
    obj_type = str(row['placetype_object']).strip()
    subj_geo = str(row['geometry_type_subject']).strip()
    obj_geo = str(row['geometry_type_object']).strip()

    if not subj_name or subj_name.lower() == 'nan' or not obj_name or obj_name.lower() == 'nan':
        continue
    
    # Ajouter les nœuds de lieux
    if subj_name not in nodes:
        nodes[subj_name] = {
            "id": subj_name,
            "label": subj_name,
            "placetype": subj_type,
            "geometry": subj_geo,
            "category": "place"
        }
    if obj_name not in nodes:
        nodes[obj_name] = {
            "id": obj_name,
            "label": obj_name,
            "placetype": obj_type,
            "geometry": obj_geo,
            "category": "place"
        }
        
    # Ajouter les nœuds de types de géométrie
    for geo in [subj_geo, obj_geo]:
        if geo and geo != 'nan':
            geo_id = f"GEO_{geo}"
            if geo_id not in nodes:
                nodes[geo_id] = {
                    "id": geo_id,
                    "label": geo,
                    "category": "geometry_type"
                }

    # Ajouter la relation spatiale principale
    links.append({
        "source": subj_name,
        "target": obj_name,
        "predicate": str(row['relation_predicate']),
        "vernacular": str(row['vernacular_relation']),
        "spatial_logic": str(row['spatial_relation']),
        "grounding_sentence": str(row['Sentence']),
        "type": "spatial"
    })
    
    # Ajouter les arcs d'attributs (Lieu -> est de type -> Géométrie)
    if subj_geo and subj_geo != 'nan':
        links.append({
            "source": subj_name,
            "target": f"GEO_{subj_geo}",
            "predicate": "is_geometry",
            "type": "ontology"
        })
    if obj_geo and obj_geo != 'nan':
        links.append({
            "source": obj_name,
            "target": f"GEO_{obj_geo}",
            "predicate": "is_geometry",
            "type": "ontology"
        })

# Déduplication des arcs (notamment pour les types de géométrie redondants)
unique_links = []
seen_links = set()
for link in links:
    link_key = (link['source'], link['target'], link.get('predicate'))
    if link_key not in seen_links:
        unique_links.append(link)
        seen_links.add(link_key)

kg_json = {
    "metadata": {
        "description": "Geographic Knowledge Graph from triplet_update_v3_70",
        "node_count": len(nodes),
        "link_count": len(unique_links)
    },
    "nodes": list(nodes.values()),
    "links": unique_links
}

# 2. Sauvegarde en fichier JSON
with open('results/knowledge_graph.json', 'w', encoding='utf-8') as f:
    json.dump(kg_json, f, indent=4, ensure_ascii=False)

print(f"KG généré : {len(nodes)} nœuds et {len(unique_links)} relations.")

