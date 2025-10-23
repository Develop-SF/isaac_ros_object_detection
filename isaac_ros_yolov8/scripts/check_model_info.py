#!/usr/bin/env python3

"""
Script to inspect YOLOv8 model files (.pt or .onnx) and extract model information.
Usage: python3 check_model_info.py /path/to/model.pt
       python3 check_model_info.py /path/to/model.onnx
"""

import argparse
import os
import sys
import yaml
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx not installed. Install with: pip install onnx")

def inspect_pt_model(model_path):
    """Inspect a YOLOv8 .pt model file."""
    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics is required for .pt files. Install with: pip install ultralytics")
        return None
    
    try:
        model = YOLO(model_path)
        num_classes = len(model.names)
        class_names = {int(k): v for k, v in model.names.items()}
        
        info = {
            'type': 'PyTorch (.pt)',
            'num_classes': num_classes,
            'class_names': class_names,
            'model_info': f"{num_classes} classes detected",
            'input_shape': 'Typically [1, 3, 640, 640] for YOLOv8',
            'output_shape': f"Typically [1, {4 + num_classes}, 8400] for YOLOv8"
        }
        return info
    except Exception as e:
        print(f"Error loading .pt model: {e}")
        return None

def inspect_onnx_model(model_path):
    """Inspect a YOLOv8 .onnx model file."""
    if not ONNX_AVAILABLE:
        print("Error: onnx is required for .onnx files. Install with: pip install onnx")
        return None
    
    try:
        model = onnx.load(model_path)
        
        # Get input/output shapes
        input_shapes = []
        output_shapes = []
        for inp in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in inp.type.tensor_type.shape.dim]
            input_shapes.append(shape)
        
        for out in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in out.type.tensor_type.shape.dim]
            output_shapes.append(shape)
        
        # For YOLOv8 detection output, typically [1, 84, 8400] where 84 = 4 (bbox) + num_classes
        num_classes = None
        class_names = {}  # Empty for ONNX
        if len(output_shapes) > 0 and len(output_shapes[0]) == 3:
            features_dim = output_shapes[0][1]
            if features_dim > 4:
                num_classes = features_dim - 4  # 4 for bbox (x,y,w,h)
                # Create template class names
                class_names = {i: f'class_{i}' for i in range(num_classes)}
        
        info = {
            'type': 'ONNX (.onnx)',
            'num_classes': num_classes,
            'class_names': class_names,
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
            'model_path': model_path
        }
        return info
    except Exception as e:
        print(f"Error loading .onnx model: {e}")
        return None

def create_config_yaml(model_path, info, config_dir):
    """Create or update the config YAML file for the model."""
    model_basename = os.path.splitext(os.path.basename(model_path))[0]
    config_filename = f"{model_basename}.yaml"
    config_path = os.path.join(config_dir, config_filename)
    
    # Check if config already exists
    if os.path.exists(config_path):
        response = input(f"Config file '{config_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print(f"Skipping config creation for {config_filename}")
            return False
    
    # Prepare YAML content
    yaml_content = {
        '#': f"Class configuration for {model_basename} model",
        'names': info['class_names']
    }
    
    # Write YAML file
    with open(config_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created/Updated config: {config_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Inspect YOLOv8 model information")
    parser.add_argument('model_path', type=str, help="Path to the model file (.pt or .onnx)")
    parser.add_argument('--no-config', action='store_true', help="Don't create config file")
    parser.add_argument('--config-dir', type=str, default=None, help="Custom config directory (default: package config)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' does not exist.")
        sys.exit(1)
    
    # Default config directory (source dir)
    if args.config_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(script_dir, '..', 'config')
    else:
        config_dir = args.config_dir
    
    # Ensure config directory exists
    os.makedirs(config_dir, exist_ok=True)
    
    file_ext = os.path.splitext(args.model_path)[1].lower()
    
    if file_ext == '.pt':
        info = inspect_pt_model(args.model_path)
    elif file_ext == '.onnx':
        info = inspect_onnx_model(args.model_path)
    else:
        print(f"Error: Unsupported file type '{file_ext}'. Use .pt or .onnx.")
        sys.exit(1)
    
    if info is None:
        print("Failed to extract model information.")
        sys.exit(1)
    
    print("\n=== Model Information ===")
    print(f"Model Type: {info['type']}")
    print(f"Model Path: {args.model_path}")
    print(f"Number of Classes: {info['num_classes']}")
    
    if isinstance(info['class_names'], dict):
        print("\nClass Names:")
        for idx, name in sorted(info['class_names'].items()):
            print(f"  {idx}: {name}")
    else:
        print(f"\nClass Names: {info['class_names']}")
    
    if 'input_shapes' in info:
        print(f"\nInput Shapes: {info['input_shapes']}")
        print(f"Output Shapes: {info['output_shapes']}")
    
    if 'model_info' in info:
        print(f"\nAdditional Model Info: {info['model_info']}")
    
    # Create config if requested
    if not args.no_config and info['num_classes'] is not None:
        create_config_yaml(args.model_path, info, config_dir)

if __name__ == "__main__":
    main()
