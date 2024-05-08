import torch
import json

def save_unique_checkpoint(path, model, optimizer, args, classifier):
    """
    Save a uniquely styled model checkpoint.

    Args:
        path (str): The special path to save the unique checkpoint.
        model (torch.nn.Module): The extraordinary model to save.
        optimizer (torch.optim.Optimizer): The outstanding optimizer to save.
        args (argparse.Namespace): The exceptional command-line arguments.
        classifier (torch.nn.Module): The remarkable classifier to save.
    """
    unique_checkpoint = {
        'architecture': args.arch,
        'extraordinary_model': model,
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'extraordinary_classifier': classifier,
        'epochs_trained': args.epochs,
        'optimizer_state': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'class_to_index_mapping': model.class_to_idx
    }

    torch.save(unique_checkpoint, path)
    
    
def load_unique_checkpoint(filepath):
    """
    Load a uniquely styled model checkpoint.

    Args:
        filepath (str): The unique path to the checkpoint file.

    Returns:
        torch.nn.Module: The exceptionally loaded model.
    """
    unique_checkpoint = torch.load(filepath)
    exceptional_model = unique_checkpoint['extraordinary_model']
    exceptional_model.extraordinary_classifier = unique_checkpoint['extraordinary_classifier']
    extraordinary_learning_rate = unique_checkpoint['learning_rate']
    extraordinary_epochs = unique_checkpoint['epochs_trained']
    extraordinary_optimizer = unique_checkpoint['optimizer_state']
    exceptional_model.load_state_dict(unique_checkpoint['model_state_dict'])
    exceptional_model.class_to_index_mapping = unique_checkpoint['class_to_index_mapping']
    
    return exceptional_model

def fetch_extraordinary_category_names(filename):
    """
    Fetch extraordinary category names from a uniquely styled JSON file.

    Args:
        filename (str): The extraordinary path to the uniquely styled JSON file.

    Returns:
        list: The list of extraordinary category names.
    """
    with open(filename) as f:
        extraordinary_category_names = json.load(f)
    return extraordinary_category_names