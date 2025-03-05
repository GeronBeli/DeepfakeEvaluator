import importlib
import os
import sys
from pathlib import Path

def list_available_models():
    """List all available model classes in the models directory."""
    model_dir = Path("models")
    models = []
    
    for file in model_dir.glob("*.py"):
        if file.name == "__init__.py" or file.name == "model_interface.py" or file.name == "model_registry.py":
            continue
        
        module_name = f"models.{file.stem}"
        try:
            module = importlib.import_module(module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr.__module__ == module.__name__:
                    models.append((file.name, attr_name))
        except Exception as e:
            print(f"Error loading module {module_name}: {e}")
    
    return models

def get_model_by_name(model_file, class_name):
    """
    Get a model class by file name and class name.
    
    This function supports models that span multiple Python files by ensuring
    that the Model directory is in the Python path, allowing imports between
    files within that directory.
    
    Args:
        model_file: Main Python file containing the model class
        class_name: Name of the model class to instantiate
        
    Returns:
        The model class that can be instantiated
    """
    if not model_file.endswith('.py'):
        model_file += '.py'
    
    model_path = os.path.join("models", model_file)
    model_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model file {model_path} does not exist")
    
    # Ensure the Model directory is in the Python path
    # This allows imports between files in the Model directory
    if model_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(model_dir))
        sys.path.insert(0, os.path.abspath(os.path.dirname(model_dir)))
    
    # Load the module
    spec = importlib.util.spec_from_file_location("module.name", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, class_name):
        raise ValueError(f"Model class {class_name} not found in {model_file}")
    
    return getattr(module, class_name)