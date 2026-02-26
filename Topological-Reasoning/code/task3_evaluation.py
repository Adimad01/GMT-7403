import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
import sys
import io

# Set UTF-8 encoding for stdout to handle special characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class SpatialRelationEvaluator:
    """
    Evaluates spatial relation predictions following the paper's methodology:
    - Conversion pairs invariant to context
    - Conversion pairs conditioned on place types or geometry types
    - Analysis with detailed tables matching paper format
    """
    
    def __init__(self, results_file: str):
        """
        Initialize evaluator with results file.
        
        Args:
            results_file: Path to CSV file with columns:
                         index, expected, predicted, match, description, model,
                         placetype_subject, geometry_type_subject,
                         placetype_object, geometry_type_object
        """
        self.df = pd.read_csv(results_file)
        self.valid_predicates = {
            "disjoint", "touches", "crosses", "within", 
            "contains", "overlaps", "equals"
        }
        
    def calculate_conversion_pair_metrics(self, description: str, expected: str) -> Dict:
        """
        Calculate frequency, accuracy, and entropy for a specific conversion pair.
        
        Args:
            description: The vernacular spatial description
            expected: The expected topological predicate
            
        Returns:
            Dictionary with metrics
        """
        # Filter for this specific conversion pair
        subset = self.df[
            (self.df['description'].str.strip() == description.strip()) & 
            (self.df['expected'] == expected) &
            (self.df['predicted'] != 'error') & 
            (self.df['predicted'] != 'skipped')
        ]
        
        if len(subset) == 0:
            return {
                'frequency': 0,
                'accuracy': 0.0,
                'entropy': 0.0,
                'total_samples': 0
            }
        
        # Frequency: count of correct predictions
        frequency = subset['match'].sum()
        
        # Accuracy: ratio of correct to total
        total = len(subset)
        accuracy = frequency / total if total > 0 else 0.0
        
        # Entropy: Shannon entropy of predictions
        predictions = subset['predicted']
        counts = predictions.value_counts()
        probabilities = counts / len(predictions)
        entropy = -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 0 else 0.0
        
        return {
            'frequency': int(frequency),
            'accuracy': float(accuracy),
            'entropy': float(entropy),
            'total_samples': int(total)
        }
    
    def analyze_invariant_pairs(self) -> pd.DataFrame:
        """
        Identify and analyze conversion pairs that consistently map to the same predicate.
        Similar to Table 8 in the paper.
        
        Returns:
            DataFrame with columns: Description, Predicate, Frequency, Accuracy, Entropy
        """
        # Group by description to find which descriptions consistently map to same predicate
        valid_df = self.df[
            (self.df['predicted'] != 'error') & 
            (self.df['predicted'] != 'skipped')
        ].copy()
        
        results = []
        
        # Group by description and expected predicate
        for (desc, expected), group in valid_df.groupby(['description', 'expected']):
            metrics = self.calculate_conversion_pair_metrics(desc, expected)
            
            # Check if this is relatively invariant (appears multiple times)
            if metrics['total_samples'] >= 5:  # At least 5 samples
                results.append({
                    'Description': desc,
                    'Predicate': expected,
                    'Frequency': metrics['frequency'],
                    'Total_Samples': metrics['total_samples'],
                    'Accuracy': metrics['accuracy'],
                    'Entropy': metrics['entropy']
                })
        
        df_results = pd.DataFrame(results)
        # Sort by accuracy (descending) and then entropy (ascending)
        if not df_results.empty:
            df_results = df_results.sort_values(['Accuracy', 'Entropy'], 
                                                ascending=[False, True])
        
        return df_results
    
    def analyze_context_conditioned_pairs(self, context_type: str = 'placetype') -> pd.DataFrame:
        """
        Analyze conversion pairs with spatial context (place type or geometry type).
        Similar to Tables 9 and 10 in the paper.
        
        Args:
            context_type: Either 'placetype' or 'geometry'
            
        Returns:
            DataFrame with context-conditioned metrics
        """
        valid_df = self.df[
            (self.df['predicted'] != 'error') & 
            (self.df['predicted'] != 'skipped')
        ].copy()
        
        results = []
        
        if context_type == 'placetype':
            # Check if placetype columns exist
            if 'placetype_subject' not in self.df.columns or 'placetype_object' not in self.df.columns:
                return pd.DataFrame()
            
            # Group by description, expected predicate, and place types
            for (desc, expected), group in valid_df.groupby(['description', 'expected']):
                # Get unique place type combinations
                for _, subgroup_df in group.groupby(['placetype_subject', 'placetype_object']):
                    if len(subgroup_df) >= 3:  # Minimum samples for meaningful analysis
                        place_context = f"{subgroup_df.iloc[0]['placetype_subject']}/{subgroup_df.iloc[0]['placetype_object']}"
                        
                        freq = subgroup_df['match'].sum()
                        total = len(subgroup_df)
                        acc = freq / total if total > 0 else 0.0
                        
                        # Calculate entropy
                        preds = subgroup_df['predicted']
                        counts = preds.value_counts()
                        probs = counts / len(preds)
                        entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
                        
                        results.append({
                            'Description': desc,
                            'Predicate': expected,
                            'Spatial_Context': place_context,
                            'Frequency': int(freq),
                            'Total_Samples': int(total),
                            'Accuracy': float(acc),
                            'Entropy': float(entropy)
                        })
        
        elif context_type == 'geometry':
            # Check if geometry columns exist
            if 'geometry_type_subject' not in self.df.columns or 'geometry_type_object' not in self.df.columns:
                return pd.DataFrame()
            
            # Group by description, expected predicate, and geometry types
            for (desc, expected), group in valid_df.groupby(['description', 'expected']):
                # Get unique geometry type combinations
                for _, subgroup_df in group.groupby(['geometry_type_subject', 'geometry_type_object']):
                    if len(subgroup_df) >= 3:  # Minimum samples
                        geom_context = f"{subgroup_df.iloc[0]['geometry_type_subject']}/{subgroup_df.iloc[0]['geometry_type_object']}"
                        
                        freq = subgroup_df['match'].sum()
                        total = len(subgroup_df)
                        acc = freq / total if total > 0 else 0.0
                        
                        # Calculate entropy
                        preds = subgroup_df['predicted']
                        counts = preds.value_counts()
                        probs = counts / len(preds)
                        entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
                        
                        results.append({
                            'Description': desc,
                            'Predicate': expected,
                            'Spatial_Context': geom_context,
                            'Frequency': int(freq),
                            'Total_Samples': int(total),
                            'Accuracy': float(acc),
                            'Entropy': float(entropy)
                        })
        
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('Accuracy', ascending=False)
        
        return df_results
    
    def analyze_description_performance(self) -> pd.DataFrame:
        """
        Analyze performance for each unique description across all contexts.
        Groups multiple instances of the same description with same expected predicate.
        
        Returns:
            DataFrame with per-description metrics
        """
        valid_df = self.df[
            (self.df['predicted'] != 'error') & 
            (self.df['predicted'] != 'skipped')
        ].copy()
        
        results = []
        
        for (desc, expected), group in valid_df.groupby(['description', 'expected']):
            freq = group['match'].sum()
            total = len(group)
            acc = freq / total if total > 0 else 0.0
            
            # Calculate entropy
            preds = group['predicted']
            counts = preds.value_counts()
            probs = counts / len(preds)
            entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
            
            # Get prediction distribution
            pred_dist = dict(counts)
            
            results.append({
                'Description': desc,
                'Expected_Predicate': expected,
                'Frequency': int(freq),
                'Total_Samples': int(total),
                'Accuracy': float(acc),
                'Entropy': float(entropy),
                'Prediction_Distribution': pred_dist
            })
        
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('Accuracy', ascending=False)
        
        return df_results
    
    def analyze_conversion_pairs(self) -> pd.DataFrame:
        """
        Analyze all conversion pairs (expected -> predicted mapping).
        
        Returns:
            DataFrame with conversion pair analysis
        """
        valid_df = self.df[
            (self.df['predicted'] != 'error') & 
            (self.df['predicted'] != 'skipped')
        ].copy()
        
        results = []
        
        for expected_rel in valid_df['expected'].unique():
            subset = valid_df[valid_df['expected'] == expected_rel]
            
            # Frequency: count of correct predictions
            correct_count = subset['match'].sum()
            
            # Accuracy: ratio of correct to total
            total_count = len(subset)
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            
            # Entropy: randomness in predictions
            predictions = subset['predicted']
            counts = predictions.value_counts()
            probabilities = counts / len(predictions)
            entropy = -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 0 else 0.0
            
            results.append({
                'expected_relation': expected_rel,
                'total_samples': total_count,
                'correct_predictions': correct_count,
                'accuracy': accuracy,
                'entropy': entropy,
                'predicted_distribution': dict(counts)
            })
        
        return pd.DataFrame(results)
    
    def analyze_by_geometry_types(self) -> pd.DataFrame:
        """
        Analyze metrics grouped by geometry type combinations.
        
        Returns:
            DataFrame with metrics per geometry type pair
        """
        # Check if geometry columns exist
        if 'geometry_type_subject' not in self.df.columns or 'geometry_type_object' not in self.df.columns:
            return pd.DataFrame()
        
        valid_df = self.df[
            (self.df['predicted'] != 'error') & 
            (self.df['predicted'] != 'skipped')
        ].copy()
        
        # Create geometry pair column
        valid_df['geometry_pair'] = (
            valid_df['geometry_type_subject'].astype(str) + 
            ' -> ' + 
            valid_df['geometry_type_object'].astype(str)
        )
        
        results = []
        
        for geom_pair in valid_df['geometry_pair'].unique():
            subset = valid_df[valid_df['geometry_pair'] == geom_pair]
            
            # Calculate metrics
            total = len(subset)
            correct = subset['match'].sum()
            accuracy = correct / total if total > 0 else 0.0
            
            # Entropy
            predictions = subset['predicted']
            counts = predictions.value_counts()
            probabilities = counts / len(predictions)
            entropy = -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 0 else 0.0
            
            results.append({
                'geometry_pair': geom_pair,
                'total_samples': total,
                'correct_predictions': correct,
                'accuracy': accuracy,
                'entropy': entropy
            })
        
        return pd.DataFrame(results)
    
    def compare_with_baseline(self, baseline_results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compare current results with baseline (e.g., without context).
        Similar to accuracy change analysis in Table 11.
        
        Args:
            baseline_results: DataFrame with baseline metrics (if None, compares with overall avg)
            
        Returns:
            DataFrame showing improvements, declines, and unchanged pairs
        """
        current = self.analyze_description_performance()
        
        if baseline_results is None:
            # Use overall average as baseline
            baseline_acc = self.df[
                (self.df['predicted'] != 'error') & 
                (self.df['predicted'] != 'skipped')
            ]['match'].mean()
            
            current['Accuracy_Change'] = current['Accuracy'] - baseline_acc
            current['Status'] = current['Accuracy_Change'].apply(
                lambda x: 'Improves' if x > 0.05 else ('Declines' if x < -0.05 else 'Unchanged')
            )
        
        return current.sort_values('Accuracy_Change', ascending=False)
    
    def generate_paper_style_tables(self, output_dir: str = "."):
        """
        Generate all tables in the paper's format.
        
        Args:
            output_dir: Directory to save output tables
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("GENERATING PAPER-STYLE EVALUATION TABLES")
        print("=" * 80)
        
        # Table 1: Invariant Conversion Pairs (like Table 8)
        print("\nTABLE 1: Conversion Pairs Invariant to Context")
        print("-" * 80)
        invariant = self.analyze_invariant_pairs()
        if not invariant.empty:
            # Format for display
            display_cols = ['Description', 'Predicate', 'Frequency', 'Accuracy', 'Entropy']
            invariant_display = invariant[display_cols].copy()
            invariant_display['Accuracy'] = invariant_display['Accuracy'].apply(lambda x: f"{x:.3f}")
            invariant_display['Entropy'] = invariant_display['Entropy'].apply(lambda x: f"{x:.3f}")
            print(invariant_display.to_string(index=False))
            invariant.to_csv(f"{output_dir}/table1_invariant_pairs.csv", index=False)
        else:
            print("No invariant pairs found (need at least 5 samples per pair)")
        
        # Table 2: Place Type Context (like Table 9)
        print("\nTABLE 2: Conversion Pairs Conditioned on Place Types")
        print("-" * 80)
        placetype = self.analyze_context_conditioned_pairs('placetype')
        if not placetype.empty:
            display_cols = ['Description', 'Predicate', 'Spatial_Context', 'Frequency', 'Accuracy', 'Entropy']
            placetype_display = placetype[display_cols].copy()
            placetype_display['Accuracy'] = placetype_display['Accuracy'].apply(lambda x: f"{x:.3f}")
            placetype_display['Entropy'] = placetype_display['Entropy'].apply(lambda x: f"{x:.3f}")
            print(placetype_display.head(20).to_string(index=False))
            if len(placetype) > 20:
                print(f"\n... and {len(placetype) - 20} more rows")
            placetype.to_csv(f"{output_dir}/table2_placetype_context.csv", index=False)
        else:
            print("No place type conditioned pairs found")
        
        # Table 3: Geometry Type Context (like Table 10)
        print("\nTABLE 3: Conversion Pairs Conditioned on Geometry Types")
        print("-" * 80)
        geometry = self.analyze_context_conditioned_pairs('geometry')
        if not geometry.empty:
            display_cols = ['Description', 'Predicate', 'Spatial_Context', 'Frequency', 'Accuracy', 'Entropy']
            geometry_display = geometry[display_cols].copy()
            geometry_display['Accuracy'] = geometry_display['Accuracy'].apply(lambda x: f"{x:.3f}")
            geometry_display['Entropy'] = geometry_display['Entropy'].apply(lambda x: f"{x:.3f}")
            print(geometry_display.head(20).to_string(index=False))
            if len(geometry) > 20:
                print(f"\n... and {len(geometry) - 20} more rows")
            geometry.to_csv(f"{output_dir}/table3_geometry_context.csv", index=False)
        else:
            print("No geometry type conditioned pairs found")
        
        # Table 4: Overall Description Performance
        print("\nTABLE 4: Overall Description Performance")
        print("-" * 80)
        desc_perf = self.analyze_description_performance()
        if not desc_perf.empty:
            display_cols = ['Description', 'Expected_Predicate', 'Frequency', 'Total_Samples', 'Accuracy', 'Entropy']
            desc_display = desc_perf[display_cols].copy()
            desc_display['Accuracy'] = desc_display['Accuracy'].apply(lambda x: f"{x:.3f}")
            desc_display['Entropy'] = desc_display['Entropy'].apply(lambda x: f"{x:.3f}")
            print(desc_display.head(30).to_string(index=False))
            if len(desc_perf) > 30:
                print(f"\n... and {len(desc_perf) - 30} more rows")
            desc_perf.to_csv(f"{output_dir}/table4_description_performance.csv", index=False)
        
        print(f"\nAll tables saved to {output_dir}/")
        print("=" * 80)
    
    def generate_summary_statistics(self) -> Dict:
        """Generate overall summary statistics."""
        valid_df = self.df[
            (self.df['predicted'] != 'error') & 
            (self.df['predicted'] != 'skipped')
        ]
        
        overall_accuracy = valid_df['match'].mean()
        
        # Per-predicate statistics
        predicate_stats = {}
        for pred in self.valid_predicates:
            pred_subset = valid_df[valid_df['expected'] == pred]
            if len(pred_subset) > 0:
                acc = pred_subset['match'].mean()
                freq = pred_subset['match'].sum()
                
                # Entropy for this predicate
                preds = pred_subset['predicted']
                counts = preds.value_counts()
                probs = counts / len(preds)
                entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
                
                predicate_stats[pred] = {
                    'frequency': int(freq),
                    'total_samples': int(len(pred_subset)),
                    'accuracy': float(acc),
                    'entropy': float(entropy)
                }
        
        return {
            'total_samples': int(len(self.df)),
            'valid_predictions': int(len(valid_df)),
            'overall_accuracy': float(overall_accuracy),
            'predicate_statistics': predicate_stats
        }
    
    def print_summary(self):
        """Print a formatted summary of key metrics."""
        summary = self.generate_summary_statistics()
        
        print("=" * 60)
        print("SPATIAL RELATION EVALUATION SUMMARY")
        print("=" * 60)
        
        # Overall metrics
        print("\nOVERALL METRICS")
        print("-" * 60)
        print(f"Total Samples:        {summary['total_samples']}")
        print(f"Valid Predictions:    {summary['valid_predictions']}")
        print(f"Overall Accuracy:     {summary['overall_accuracy']:.2%}")
        
        # Frequency
        print("\nFREQUENCY (Correct Predictions per Predicate)")
        print("-" * 60)
        freq_dict = {pred: stats['frequency'] for pred, stats in summary['predicate_statistics'].items()}
        for pred, count in sorted(freq_dict.items()):
            print(f"{pred:15s}: {count}")
        
        # Per-relation accuracy
        print("\nACCURACY BY EXPECTED RELATION")
        print("-" * 60)
        for pred, stats in sorted(summary['predicate_statistics'].items()):
            print(f"{pred:15s}: {stats['accuracy']:.2%}")
        
        # Per-relation entropy
        print("\nENTROPY BY EXPECTED RELATION")
        print("-" * 60)
        print("(Lower entropy = more consistent predictions)")
        for pred, stats in sorted(summary['predicate_statistics'].items()):
            print(f"{pred:15s}: {stats['entropy']:.4f}")
        
        print("\n" + "=" * 60)
    
    def generate_full_report(self, output_file: str = None) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_file: Optional path to save JSON report
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        def convert_to_native_types(obj):
            """Convert numpy/pandas types to native Python types for JSON serialization."""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            return obj
        
        summary = self.generate_summary_statistics()
        
        report = {
            'overall_metrics': {
                'total_samples': summary['total_samples'],
                'valid_predictions': summary['valid_predictions'],
                'accuracy': summary['overall_accuracy']
            },
            'predicate_statistics': convert_to_native_types(summary['predicate_statistics']),
            'conversion_pair_analysis': convert_to_native_types(
                self.analyze_conversion_pairs().to_dict('records')
            ),
            'geometry_type_analysis': convert_to_native_types(
                self.analyze_by_geometry_types().to_dict('records')
            )
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Full report saved to: {output_file}")
        
        return report
    
    def save_complete_report_to_txt(self, output_file: str):
        """
        Save a complete text report with all analyses.
        
        Args:
            output_file: Path to save the text report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE SPATIAL RELATION EVALUATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary Statistics
            summary = self.generate_summary_statistics()
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples:        {summary['total_samples']}\n")
            f.write(f"Valid Predictions:    {summary['valid_predictions']}\n")
            f.write(f"Overall Accuracy:     {summary['overall_accuracy']:.2%}\n\n")
            
            # Per-predicate statistics
            f.write("PER-PREDICATE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Predicate':<15} {'Frequency':<12} {'Total':<8} {'Accuracy':<12} {'Entropy':<10}\n")
            f.write("-" * 80 + "\n")
            for pred, stats in sorted(summary['predicate_statistics'].items()):
                f.write(f"{pred:<15} {stats['frequency']:<12} {stats['total_samples']:<8} "
                       f"{stats['accuracy']:<12.2%} {stats['entropy']:<10.4f}\n")
            f.write("\n")
            
            # Conversion Pair Analysis
            f.write("CONVERSION PAIR ANALYSIS\n")
            f.write("-" * 80 + "\n")
            conversion = self.analyze_conversion_pairs()
            f.write(conversion.to_string(index=False))
            f.write("\n\n")
            
            # Geometry Type Analysis
            f.write("GEOMETRY TYPE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            geometry = self.analyze_by_geometry_types()
            f.write(geometry.to_string(index=False))
            f.write("\n\n")
            
            # Description Performance
            f.write("DESCRIPTION PERFORMANCE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            desc_perf = self.analyze_description_performance()
            display_cols = ['Description', 'Expected_Predicate', 'Frequency', 'Total_Samples', 'Accuracy', 'Entropy']
            f.write(desc_perf[display_cols].to_string(index=False))
            f.write("\n\n")
            
            # Paper-style analyses
            f.write("=" * 80 + "\n")
            f.write("PAPER-STYLE DETAILED ANALYSES\n")
            f.write("=" * 80 + "\n\n")
            
            # Invariant pairs
            f.write("TABLE 1: Conversion Pairs Invariant to Context\n")
            f.write("-" * 80 + "\n")
            invariant = self.analyze_invariant_pairs()
            if not invariant.empty:
                f.write(invariant.to_string(index=False))
            else:
                f.write("No invariant pairs found (need at least 5 samples per pair)\n")
            f.write("\n\n")
            
            # Place type context
            f.write("TABLE 2: Conversion Pairs Conditioned on Place Types\n")
            f.write("-" * 80 + "\n")
            placetype = self.analyze_context_conditioned_pairs('placetype')
            if not placetype.empty:
                f.write(placetype.to_string(index=False))
            else:
                f.write("No place type conditioned pairs found\n")
            f.write("\n\n")
            
            # Geometry type context
            f.write("TABLE 3: Conversion Pairs Conditioned on Geometry Types\n")
            f.write("-" * 80 + "\n")
            geometry_context = self.analyze_context_conditioned_pairs('geometry')
            if not geometry_context.empty:
                f.write(geometry_context.to_string(index=False))
            else:
                f.write("No geometry type conditioned pairs found\n")
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Complete text report saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    
    print('\n')
    print('='*60)
    # ---------------------------------------------------------------
    print('EVALUATION OF GPTOSS FEW-SHOT RESULTS')
    print('='*60)
    print('\n')
    
    # Initialize evaluator with your results file
    evaluator_gptoss = SpatialRelationEvaluator("./results/gptoss_live_log_few_shot.txt")
    
    # Print summary
    evaluator_gptoss.print_summary()
    
    # Generate and save full report (JSON)
    report_gptoss = evaluator_gptoss.generate_full_report("./results/evaluation_report_gptoss_few_shot.json")
    
    # Save complete text report
    evaluator_gptoss.save_complete_report_to_txt("./results/evaluation_report_gptoss_few_shot.txt")
    
    # Generate paper-style tables
    evaluator_gptoss.generate_paper_style_tables(output_dir="./results/gptoss_results_few_shot")

    print('\n')
    print('='*60)
    # ---------------------------------------------------------------
    print('EVALUATION OF GPTOSS ZERO-SHOT RESULTS')
    print('='*60)
    print('\n')
    
    # Initialize evaluator with your results file
    evaluator_gptoss = SpatialRelationEvaluator("./results/gptoss_live_log_zero_shot.txt")
    
    # Print summary
    evaluator_gptoss.print_summary()
    
    # Generate and save full report (JSON)
    report_gptoss = evaluator_gptoss.generate_full_report("./results/evaluation_report_gptoss_zero_shot.json")
    
    # Save complete text report
    evaluator_gptoss.save_complete_report_to_txt("./results/evaluation_report_gptoss_zero_shot.txt")
    
    # Generate paper-style tables
    evaluator_gptoss.generate_paper_style_tables(output_dir="./results/gptoss_results_zero_shot")