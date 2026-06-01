import json
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define your directories
input_dir = 'results'
output_dir = 'confusion_matrix'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all JSON files in the results folder
json_files = glob.glob(os.path.join(input_dir, '*.json'))

if not json_files:
    print(f"No JSON files found in the '{input_dir}' directory.")

for file in json_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        if not results:
            continue
            
        y_true = [r['expected'] for r in results]
        y_pred = [r['predicted'] for r in results]
        
        # Get unique labels from both true and predicted to ensure all are shown
        labels = sorted(list(set(y_true + y_pred)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
        
        # Extract filename and clean up the title
        basename = os.path.basename(file)
        title = basename.replace('_ckpt.json', '').replace('voletc_', '').upper()
        ax.set_title(f"Confusion Matrix: {title}")
        
        plt.tight_layout()
        
        # Construct output path in the new directory
        output_filename = os.path.join(output_dir, basename.replace('.json', '_confusion_matrix.png'))
        plt.savefig(output_filename)
        plt.close(fig)
        
        print(f"Saved {output_filename}")
        
    except Exception as e:
        print(f"Error processing {file}: {e}")