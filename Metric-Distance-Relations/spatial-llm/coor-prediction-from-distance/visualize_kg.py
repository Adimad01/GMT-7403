import json
import os
import networkx as nx
from pyvis.network import Network

def visualize_json_graph(json_path: str, output_path: str, max_nodes: int = 100):
    print(f"Loading JSON graph from: {json_path}")
    
    # 1. Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Found {data['metadata']['node_count']} nodes and {data['metadata']['edge_count']} edges.")

    # 2. Rebuild the NetworkX Graph
    G = nx.Graph()
    
    # Add Nodes
    for node in data["nodes"]:
        hover_info = f"<b>{node['name']}</b><br>State: {node['state']}"
        
        G.add_node(
            node["id"], 
            title=hover_info, 
            group=node["state"],
            label=node["name"].split(",")[0],
            
            # 👇 EXTRA DATA: We pass these so the JavaScript sidebar can access them!
            full_name=node["name"],
            state=node["state"],
            mentions=node["corpus_mentions"],
            lat=node["lat"],
            lon=node["lon"]
        )

    # Add Edges
    for link in data["links"]:
        hover_info = f"Distance: {link['distance']:.1f} km"
        
        G.add_edge(
            link["source"], 
            link["target"], 
            title=hover_info,
            
            # 👇 EXTRA DATA for the edge click event (ADDED ALL SIGNALS HERE)
            distance=link['distance'],
            co_occ=link['corpus_signals']['co_occ_count'],
            near_count=link['corpus_signals']['near_count'],
            close_count=link['corpus_signals']['close_count'],
            far_count=link['corpus_signals']['far_count'],
            and_count=link['corpus_signals']['and_count']
        )

    # 3. Filter to a Safe Subgraph
    sorted_cities = sorted(data["nodes"], key=lambda x: x["corpus_mentions"], reverse=True)
    top_city_ids = [city["id"] for city in sorted_cities[:max_nodes]]
    
    subG = G.subgraph(top_city_ids)
    print(f"Created a safe subgraph with the top {subG.number_of_nodes()} cities.")

    # 4. Render with PyVis
    net = Network(height="100vh", width="100%", bgcolor="#fcfcfc", font_color="#333333", cdn_resources="remote")
    net.from_nx(subG)
    net.show_buttons(filter_=["physics"])
    
    # Save the base HTML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net.write_html(output_path)
    
    # ==========================================================
    # 5. THE MAGIC: INJECT CUSTOM SIDEBAR & JAVASCRIPT
    # ==========================================================
    with open(output_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # The HTML/CSS for the Sidebar + The JavaScript to listen for clicks
    sidebar_injection = """
    <div id="custom-sidebar" style="position: absolute; top: 0; right: 0; width: 350px; height: 100vh; background: #ffffff; border-left: 2px solid #e0e0e0; padding: 25px; box-sizing: border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; z-index: 9999; box-shadow: -4px 0 15px rgba(0,0,0,0.05); overflow-y: auto;">
        <h2 style="margin-top: 0; padding-bottom: 15px; border-bottom: 2px solid #333; color: #222;">Information Panel</h2>
        <div id="info-content" style="font-size: 15px; line-height: 1.6; color: #444;">
            <p style="color: #888; font-style: italic; margin-top: 20px;">👈 Click on any node or edge in the graph to see its detailed geospatial data here.</p>
        </div>
    </div>

    <script type="text/javascript">
        window.addEventListener('load', function() {
            setTimeout(function() {
                if (typeof network !== 'undefined') {
                    
                    // Shrink the main network canvas so it doesn't hide behind the sidebar
                    var container = document.getElementById('mynetwork');
                    if(container) {
                        container.style.width = 'calc(100% - 350px)';
                    }
                    
                    // Listen for clicks on the network
                    network.on("click", function (params) {
                        var content = document.getElementById("info-content");
                        
                        // 1. If a Node is clicked
                        if (params.nodes.length > 0) {
                            var nodeId = params.nodes[0];
                            var nodeData = nodes.get(nodeId); 
                            
                            if (nodeData) {
                                var html = "<h3 style='color: #0056b3; margin-bottom: 10px;'>" + nodeData.full_name + "</h3>";
                                html += "<b>State:</b> " + nodeData.state + "<br>";
                                html += "<b>Corpus Mentions:</b> " + nodeData.mentions.toLocaleString() + "<br>";
                                html += "<b>Latitude:</b> " + nodeData.lat + "°<br>";
                                html += "<b>Longitude:</b> " + nodeData.lon + "°<br>";
                                
                                // Generate a quick Google Maps link
                                var mapsLink = "https://www.google.com/maps/search/?api=1&query=" + nodeData.lat + "," + nodeData.lon;
                                html += "<br><a href='" + mapsLink + "' target='_blank' style='display: inline-block; margin-top: 10px; padding: 8px 15px; background: #28a745; color: white; text-decoration: none; border-radius: 4px; font-weight: bold;'>📍 View on Map</a>";
                                
                                content.innerHTML = html;
                            }
                        } 
                        // 2. If an Edge is clicked (and no node is clicked)
                        else if (params.edges.length > 0) {
                            var edgeId = params.edges[0];
                            var edgeData = edges.get(edgeId);
                            
                            if (edgeData) {
                                var html = "<h3 style='color: #d35400; margin-bottom: 10px;'>Distance & Signals</h3>";
                                html += "<b>Direct Distance:</b> " + edgeData.distance.toFixed(2) + " km<br><br>";
                                
                                // ADDED NEW CORPUS SIGNALS SECTION HERE 👇
                                html += "<h4 style='color: #444; margin-bottom: 5px; margin-top: 5px; border-bottom: 1px solid #eee; padding-bottom: 3px;'>Corpus Co-occurrences</h4>";
                                html += "<b>Total Mentions:</b> " + edgeData.co_occ + "<br>";
                                html += "<b>'Near' Context:</b> " + (edgeData.near_count || 0) + "<br>";
                                html += "<b>'Close' Context:</b> " + (edgeData.close_count || 0) + "<br>";
                                html += "<b>'Far' Context:</b> " + (edgeData.far_count || 0) + "<br>";
                                html += "<b>'And' Context:</b> " + (edgeData.and_count || 0) + "<br>";
                                
                                content.innerHTML = html;
                            }
                        } 
                        // 3. If empty space is clicked
                        else {
                            content.innerHTML = "<p style='color: #888; font-style: italic; margin-top: 20px;'>👈 Click on any node or edge in the graph to see its detailed geospatial data here.</p>";
                        }
                    });
                }
            }, 800); // Wait 800ms to ensure the network has finished loading
        });
    </script>
    </body>
    """
    
    # Replace the closing body tag with our custom injected code
    final_html = html_content.replace("</body>", sidebar_injection)
    
    # Write it back to the file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"\n✅ Visualization saved to: {output_path}")
    print("👉 Open the HTML file in Chrome/Safari to interact with it.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(base_dir, "outputs", "geospatial_knowledge_graph_final.json")
    html_output = os.path.join(base_dir, "outputs", "interactive_kg.html")
    
    if os.path.exists(json_file):
        visualize_json_graph(json_file, html_output, max_nodes=12)
    else:
        print(f"❌ Error: Could not find {json_file}.")
        print("Run 'python3 geographic_kg.py' first to generate the JSON.")