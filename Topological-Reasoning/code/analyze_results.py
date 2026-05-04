import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
RESULTS_DIR = "./results/"

def generate_confusion_matrix(df, filename, directory):
    """
    Generates and saves a confusion matrix plot for a single log file.
    """
    # Filter out 'invalid' or nulls if you only want to compare actual classes
    # or keep them to see how often the model fails to produce a valid label.
    y_true = df['expected'].astype(str).str.lower().str.strip()
    y_pred = df['predicted'].astype(str).str.lower().str.strip()

    # Get unique labels sorted alphabetically
    labels = sorted(list(set(y_true) | set(y_pred)))

    # Compute Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {filename}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    plot_path = os.path.join(directory, f"cm_{filename.replace('.txt', '.png')}")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def evaluate_all_logs(directory):
    """Iterate over all .txt log files in a directory, compute metrics, and save confusion matrices."""
    search_pattern = os.path.join(directory, "*.txt")
    log_files = glob.glob(search_pattern)
    
    if not log_files:
        print(f"No .txt files found in {directory}")
        return

    print(f"Found {len(log_files)} log files. Evaluating...\n")
    print("-" * 75)
    
    summary_data = []

    for filepath in log_files:
        filename = os.path.basename(filepath)
        
        try:
            df = pd.read_csv(filepath, on_bad_lines='skip')
            df.columns = df.columns.str.strip()
            
            missing_cols = {'expected', 'predicted', 'match'} - set(df.columns)
            if missing_cols:
                print(f"⚠️  SKIPPING {filename}: missing columns {missing_cols}. Found: {list(df.columns)}")
                continue

            # Standardize data for evaluation
            df['match'] = df['match'].astype(str).str.lower() == 'true'
            df['predicted'] = df['predicted'].astype(str).str.lower().str.strip()
            df['expected'] = df['expected'].astype(str).str.lower().str.strip()

            total_samples = len(df)
            correct_matches = df['match'].sum()
            accuracy = (correct_matches / total_samples) * 100 if total_samples > 0 else 0
            invalid_count = (df['predicted'] == 'invalid').sum()
            invalid_rate = (invalid_count / total_samples) * 100 if total_samples > 0 else 0

            # --- NEW: Generate Confusion Matrix ---
            cm_path = generate_confusion_matrix(df, filename, directory)

            summary_data.append({
                "File": filename,
                "Total": total_samples,
                "Accuracy (%)": round(accuracy, 2),
                "Invalid (%)": round(invalid_rate, 2)
            })
            
            print(f"📄 {filename}")
            print(f"   Accuracy:  {accuracy:.2f}%")
            print(f"   CM Saved:  {cm_path}")
            print("-" * 75)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n📊 FINAL SUMMARY TABLE")
        print("=" * 75)
        print(summary_df.to_string(index=False))
        
        summary_out_path = os.path.join(directory, "evaluation_summary.csv")
        summary_df.to_csv(summary_out_path, index=False)
        print(f"\nSummary saved to: {summary_out_path}")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory not found: {RESULTS_DIR}")
    else:
        evaluate_all_logs(RESULTS_DIR)