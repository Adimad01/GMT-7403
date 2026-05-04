#!/usr/bin/env python3
"""
Geospatial Model Evaluation Script
Calculates Haversine distances, generates a text report, and produces
high-impact visualizations (CDF, Log-Boxplots, Bar Charts, and Geo-Maps).
"""

import json
import os
import glob
import re
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Attempt to import geopandas for the map visualization
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# --- Configuration ---
OUTPUTS_DIR = "outputs"
EARTH_RADIUS_KM = 6371.0

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = EARTH_RADIUS_KM
    return c * r

def extract_coordinates(text):
    """
    Robustly extract Latitude and Longitude from raw text.
    Handles standard '-29.13, -65.21' and cardinal '34.0°S, 60.0°W'.
    """
    if not isinstance(text, str):
        return None, None

    # Strategy 1: Standard float comma-separated
    m_std = re.search(r'(-?\d{1,3}\.\d+)[^\d\-a-zA-Z]*?,\s*(-?\d{1,3}\.\d+)', text)
    if m_std:
        try:
            lat, lon = float(m_std.group(1)), float(m_std.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass

    # Strategy 2: Cardinal directions
    m_card = re.search(r'(\d{1,3}\.\d+)[^\w]*([NSns])\s*,\s*(\d{1,3}\.\d+)[^\w]*([EWew])', text)
    if m_card:
        try:
            lat_val = float(m_card.group(1))
            lat_dir = m_card.group(2).upper()
            lon_val = float(m_card.group(3))
            lon_dir = m_card.group(4).upper()
            
            lat = lat_val if lat_dir == 'N' else -lat_val
            lon = lon_val if lon_dir == 'E' else -lon_val
            
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass

    return None, None

def evaluate_file(filepath):
    """Evaluate a single JSON file and return aggregate metrics + raw data for plotting."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = json.load(f)
        
    data = content.get('data', content) if isinstance(content, dict) else content
    if not data:
        return None

    errors_km = []
    geo_vectors = []
    valid_preds = 0
    total_samples = len(data)

    for item in data:
        true_lat = item.get('Latitude')
        true_lon = item.get('Longitude')
        
        raw_text = str(item.get('output', '')) + " " + str(item.get('raw_output', ''))
        pred_lat, pred_lon = extract_coordinates(raw_text)
        
        if pred_lat is not None and pred_lon is not None and true_lat is not None and true_lon is not None:
            dist_error = haversine(true_lat, true_lon, pred_lat, pred_lon)
            errors_km.append(dist_error)
            valid_preds += 1
            
            geo_vectors.append({
                'true_lat': true_lat, 'true_lon': true_lon,
                'pred_lat': pred_lat, 'pred_lon': pred_lon,
                'error_km': dist_error
            })

    if valid_preds == 0:
        return {
            'metrics': {'total': total_samples, 'valid_rate': 0.0, 'mean_km': float('inf'), 'median_km': float('inf'), 'rmse_km': float('inf'), 'acc_100km': 0.0},
            'raw_errors': [], 'geo_vectors': []
        }

    errors_array = np.array(errors_km)
    metrics = {
        'total': total_samples,
        'valid_rate': (valid_preds / total_samples) * 100,
        'mean_km': np.mean(errors_array),
        'median_km': np.median(errors_array),
        'rmse_km': np.sqrt(np.mean(errors_array**2)),
        'acc_100km': np.mean(errors_array <= 100.0) * 100
    }
    
    return {'metrics': metrics, 'raw_errors': errors_km, 'geo_vectors': geo_vectors}

def generate_visualizations(summary_data, raw_error_data, geo_data, output_dir):
    """Generates all four high-impact geospatial visualizations."""
    if not summary_data: return
    
    print("\n📊 Generating Visualizations...")
    sns.set_theme(style="whitegrid")
    
    # Convert lists to DataFrames
    df_sum = pd.DataFrame(summary_data)
    df_sum['Config'] = df_sum['model'] + " | " + df_sum['p_type'] + " | " + df_sum['shots']
    
    df_raw = pd.DataFrame(raw_error_data)
    
    # ---------------------------------------------------------
    # 1. Formatting Compliance (Bar Chart)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_sum, x='Config', y='valid_rate', palette='viridis')
    plt.title("Coordinate Formatting Compliance by Model Configuration", fontsize=14, fontweight='bold')
    plt.ylabel("Valid Formatting Rate (%)", fontsize=12)
    plt.xlabel("Model Configuration", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CHART_1_Formatting_Compliance.png"), dpi=300)
    plt.close()

    if df_raw.empty:
        print("⚠️ Not enough valid predictions to generate distance charts.")
        return

    # ---------------------------------------------------------
    # 2. CDF of Spatial Error
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=df_raw, x='error_km', hue='Config', linewidth=2.5)
    plt.title("Cumulative Distribution Function (CDF) of Distance Error", fontsize=14, fontweight='bold')
    plt.ylabel("Proportion of Predictions", fontsize=12)
    plt.xlabel("Distance Error (km)", fontsize=12)
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.5, label='100km Threshold')
    plt.xlim(0, 2000) # Truncate tail for readability
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CHART_2_Error_CDF.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Log-Scaled Box Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_raw, x='Config', y='error_km', palette='Set2')
    plt.yscale('log') # Log scale is crucial for geo-errors
    plt.title("Distribution of Distance Errors (Log Scale)", fontsize=14, fontweight='bold')
    plt.ylabel("Error in Kilometers (Log Scale)", fontsize=12)
    plt.xlabel("Model Configuration", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100km Threshold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CHART_3_Log_Boxplot.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. Geospatial Error Vector Map (Upgraded with Insights)
    # ---------------------------------------------------------
    if HAS_GEOPANDAS and geo_data:
        try:
            from matplotlib.lines import Line2D
            
            fig, ax = plt.subplots(figsize=(16, 9))
            
            # Load world map with a cleaner, softer palette
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            world.plot(ax=ax, color='#f4f4f2', edgecolor='#c0c0c0', linewidth=0.5)
            
            # Identify the best configuration to plot
            best_stats = df_sum.loc[df_sum['rmse_km'].idxmin()]
            best_config = best_stats['Config']
            best_geo = [g for g in geo_data if g['Config'] == best_config]
            
            # Sample to prevent overwhelming the plot (optional, adjustable)
            if len(best_geo) > 400: best_geo = random.sample(best_geo, 400)
            
            # Sort by error ascending so massive red errors are drawn ON TOP of small green ones
            best_geo.sort(key=lambda x: x['error_km'])
            
            for item in best_geo:
                err = item['error_km']
                
                # Color and thickness logic based on error severity
                if err <= 100:
                    line_color, alpha, lw = 'mediumseagreen', 0.6, 1.0
                elif err <= 500:
                    line_color, alpha, lw = 'darkorange', 0.7, 1.5
                else:
                    line_color, alpha, lw = 'crimson', 0.85, 2.0
                
                # 1. Draw Error Vector (Line)
                ax.plot([item['true_lon'], item['pred_lon']], [item['true_lat'], item['pred_lat']], 
                        color=line_color, alpha=alpha, linewidth=lw, zorder=2)
                
                # 2. Draw Predicted Location (Red Cross)
                ax.plot(item['pred_lon'], item['pred_lat'], marker='x', color='crimson', 
                        markersize=5, markeredgewidth=1.5, alpha=0.8, zorder=3)
                
                # 3. Draw True Location (Green Dot with border for pop)
                ax.plot(item['true_lon'], item['true_lat'], marker='o', color='mediumseagreen', 
                        markersize=5, markeredgecolor='black', markeredgewidth=0.5, alpha=0.9, zorder=4)
            
            # --- Add Custom Legend ---
            custom_lines = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumseagreen', markeredgecolor='black', markersize=8, label='Actual Location'),
                Line2D([0], [0], marker='x', color='w', markeredgecolor='crimson', markersize=8, markeredgewidth=2, label='Predicted Location'),
                Line2D([0], [0], color='mediumseagreen', lw=2, label='Excellent (≤ 100km)'),
                Line2D([0], [0], color='darkorange', lw=2, label='Moderate (100 - 500km)'),
                Line2D([0], [0], color='crimson', lw=2, label='Severe (> 500km)')
            ]
            ax.legend(handles=custom_lines, loc='lower left', framealpha=0.95, 
                      title="Spatial Key", title_fontsize='13', fontsize='11', edgecolor='gray')

            # --- Add Insights Text Box (Top Left) ---
            insight_text = (
                f"MODEL PERFORMANCE SUMMARY\n"
                f"-------------------------\n"
                f"Model:  {best_stats['model']}\n"
                f"Setup:  {best_stats['p_type']} | {best_stats['shots']}\n\n"
                f"Valid Format:   {best_stats['valid_rate']:.1f}%\n"
                f"Mean Error:     {best_stats['mean_km']:,.1f} km\n"
                f"Median Error:   {best_stats['median_km']:,.1f} km\n"
                f"Accuracy <100km:{best_stats['acc_100km']:.1f}%\n"
                f"Samples Plotted:{len(best_geo)}"
            )
            props = dict(boxstyle='round,pad=0.6', facecolor='#ffffff', alpha=0.95, edgecolor='#aaaaaa')
            ax.text(0.02, 0.97, insight_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props, family='monospace', zorder=5)

            # Map Styling
            plt.title(f"Geospatial Error Vectors & Severity Tracking", fontsize=16, fontweight='bold', pad=15)
            plt.xlim(-180, 180)
            plt.ylim(-90, 90)
            
            # Remove lat/lon axis ticks for a cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "CHART_4_Geo_Vector_Map_Upgraded.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Upgraded Insights Map Visualization generated.")
        except Exception as e:
            print(f"⚠️ Could not generate map: {e}")
    else:
        print("⚠️ Skipping map visualization (Geopandas not installed or no geo data).")
        
    print("✅ All statistical charts generated and saved to outputs/")

def main():
    """Scan output JSON files, compute metrics, and generate all visualizations."""
    print(f"Scanning for JSON files in {OUTPUTS_DIR}/...\n")
    
    # Grab all results JSON files
    files = glob.glob(os.path.join(OUTPUTS_DIR, "results_gptoss-*.json"))
    
    if not files:
        print("No files found to evaluate.")
        return

    summary_data = []
    raw_error_data = []
    geo_data = []
    
    for f in files:
        filename = os.path.basename(f)
        
        # Parse metadata from filename: results_gptoss-base_3-shot_type0_20260326.json
        match = re.search(r'results_(gptoss-(?:base|lora))_(zero-shot|3-shot)_type(\d+)_', filename)
        if not match:
            continue
            
        model_name, shots, p_type = match.groups()
        config_label = f"{model_name} | Type {p_type} | {shots}"
        
        results = evaluate_file(f)
        if results:
            # 1. Store Summary Metrics
            summary_data.append({
                'model': model_name,
                'shots': shots,
                'p_type': f"Type {p_type}",
                **results['metrics']
            })
            
            # 2. Store Raw Errors for CDF & Boxplot
            for err in results['raw_errors']:
                raw_error_data.append({'Config': config_label, 'error_km': err})
                
            # 3. Store Geo Vectors for Map
            for g in results['geo_vectors']:
                g['Config'] = config_label
                geo_data.append(g)

    if not summary_data:
        print("No valid data extracted.")
        return

    # Sort by Model -> Type -> Shots
    summary_data.sort(key=lambda x: (x['model'], x['p_type'], x['shots']))

    # Print Unified Table
    print("=" * 105)
    print(f"{'Model':<15} | {'Prompt':<7} | {'Shots':<10} | {'Valid %':<8} | {'Mean (km)':<10} | {'Median (km)':<11} | {'RMSE (km)':<10}")
    print("-" * 105)
    for row in summary_data:
        print(f"{row['model']:<15} | {row['p_type']:<7} | {row['shots']:<10} | {row['valid_rate']:>7.1f}% | {row['mean_km']:>10.1f} | {row['median_km']:>11.1f} | {row['rmse_km']:>10.1f}")
    print("=" * 105)

    # Generate Graphs
    generate_visualizations(summary_data, raw_error_data, geo_data, OUTPUTS_DIR)

if __name__ == "__main__":
    main()