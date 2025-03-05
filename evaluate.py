import os
import argparse
import torch
import pandas as pd
import numpy as np
import importlib.util
import sys
from pathlib import Path

def load_model_class(model_file, model_class_name):
    """
    Dynamically load a model class from a file.
    Supports models that span multiple Python files.
    """
    if not model_file.endswith('.py'):
        model_file += '.py'
    
    model_path = os.path.abspath(model_file)
    model_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model file {model_path} does not exist")
    
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
        sys.path.insert(0, os.path.dirname(model_dir))
    
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, model_class_name):
        raise ValueError(f"Model class {model_class_name} not found in {model_file}")
    
    return getattr(module, model_class_name)

def evaluate_model(model_file, model_class_name, weights_path, data_dir, output_csv):
    """Evaluate a deepfake detection model and save results to CSV."""
    
    # Load model class and instantiate
    ModelClass = load_model_class(model_file, model_class_name)
    model = ModelClass()
    
    # Load weights
    model.load_state_dict(weights_path)
    
    # Import dataloader
    # We need to ensure Data directory is in the path
    data_dir_path = os.path.abspath("Data")
    if data_dir_path not in sys.path:
        sys.path.insert(0, data_dir_path)
        sys.path.insert(0, os.path.dirname(data_dir_path))
    
    from data.dataloader import get_dataloader
    
    # Create dataloader
    dataloader = get_dataloader(data_dir)
    
    # Evaluation
    results = []
    
    for batch in dataloader:
        image = batch['image']
        label = batch['label'].item()
        path = batch['path'][0]
        
        # Make prediction
        pred_label, pred_value = model.predict(image)
        
        # Store result
        results.append({
            'FilePath': path,
            'CorrectLabel': 'REAL' if label == 0 else 'FAKE',
            'PredictedLabel': 'REAL' if pred_label == 0 else 'FAKE',
            'PredictedValue': pred_value
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Calculate metrics
    correct = df['CorrectLabel'] == df['PredictedLabel']
    accuracy = correct.mean()
    
    # Calculate confusion matrix
    real_correct = ((df['CorrectLabel'] == 'REAL') & (df['PredictedLabel'] == 'REAL')).sum()
    real_total = (df['CorrectLabel'] == 'REAL').sum()
    fake_correct = ((df['CorrectLabel'] == 'FAKE') & (df['PredictedLabel'] == 'FAKE')).sum()
    fake_total = (df['CorrectLabel'] == 'FAKE').sum()

    real_detection_rate = real_correct / real_total if real_total > 0 else 0
    fake_detection_rate = fake_correct / fake_total if fake_total > 0 else 0
    
    print(f"Evaluation Results:")
    print(f"Total images: {len(df)}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Real Detection Rate: {real_detection_rate:.4f} ({real_detection_rate})")
    print(f"Fake Detection Rate: {fake_detection_rate:.4f} ({fake_detection_rate})")
    print(f"Results saved to {output_csv}")




def main():
    parser = argparse.ArgumentParser(description='Evaluate a deepfake detection model')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the model file')
    parser.add_argument('--model_class', type=str, required=True,
                        help='Name of the model class in the model file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the model weights file')
    parser.add_argument('--data_dir', type=str, default='sample_images',
                        help='Directory containing REAL and FAKE subdirectories')
    parser.add_argument('--output', type=str, default='.\\evaluation_results.csv',
                        help='Path to save the evaluation results CSV')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('weights', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs(os.path.join('sample_images', 'REAL'), exist_ok=True)
    os.makedirs(os.path.join('sample_images', 'FAKE'), exist_ok=True)
    
    evaluate_model(
        args.model,
        args.model_class,
        args.weights,
        args.data_dir,
        args.output
    )

if __name__ == "__main__":
    main()