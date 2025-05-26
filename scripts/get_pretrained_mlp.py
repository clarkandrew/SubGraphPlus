#!/usr/bin/env python3
"""
Script to download or create pre-trained MLP model for SubgraphRAG+
"""

import os
import sys
import logging
import argparse
import requests
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)

class SimpleMLP(nn.Module):
    """Simple MLP for SubgraphRAG scoring - matches the expected architecture"""
    def __init__(self, input_dim=4116, hidden_dim=1024, output_dim=1):
        super(SimpleMLP, self).__init__()
        # Architecture must match the saved model: pred.0 (input -> hidden), pred.2 (hidden -> output)
        self.pred = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # pred.0
            nn.ReLU(),                           # pred.1 (no parameters)
            nn.Linear(hidden_dim, output_dim)    # pred.2
        )
    
    def forward(self, x):
        return self.pred(x)


def download_pretrained_model(url: str, model_path: Path) -> bool:
    """
    Download pre-trained MLP model from URL
    
    Args:
        url: URL to download the model from
        model_path: Path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        console.print(f"[blue]Downloading pre-trained MLP model from {url}...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=None)
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Ensure parent directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            progress.update(task, completed=True)
        
        console.print(f"[green]✅ Successfully downloaded model to {model_path}[/green]")
        return True
        
    except requests.RequestException as e:
        console.print(f"[red]❌ Download failed: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Error downloading model: {e}[/red]")
        return False


def create_placeholder_model(model_path: Path, input_dim: int = 4116, hidden_dim: int = 1024) -> bool:
    """
    Create a placeholder MLP model for development/demo purposes
    
    Args:
        model_path: Path to save the model
        input_dim: Input dimension for the model
        hidden_dim: Hidden layer dimension
        
    Returns:
        True if successful, False otherwise
    """
    try:
        console.print(f"[yellow]Creating placeholder MLP model...[/yellow]")
        
        # Create model with standard dimensions
        model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1)
        
        # Initialize with reasonable weights
        with torch.no_grad():
            # Initialize first layer with Xavier uniform
            nn.init.xavier_uniform_(model.pred[0].weight)
            nn.init.zeros_(model.pred[0].bias)
            
            # Initialize second layer with smaller weights
            nn.init.xavier_uniform_(model.pred[2].weight, gain=0.1)
            nn.init.zeros_(model.pred[2].bias)
        
        # Save the model state dict in the expected format
        checkpoint = {
            'config': {
                'input_dim': input_dim, 
                'hidden_dim': hidden_dim, 
                'output_dim': 1,
                'model_type': 'SimpleMLP',
                'created_by': 'get_pretrained_mlp.py',
                'note': 'Placeholder model for development/demo purposes'
            },
            'model_state_dict': model.state_dict()
        }
        
        # Ensure the models directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, model_path)
        console.print(f"[green]✅ Created placeholder MLP model at {model_path}[/green]")
        console.print(f"[dim]   Input dim: {input_dim}, Hidden dim: {hidden_dim}[/dim]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error creating placeholder model: {e}[/red]")
        return False


def verify_model(model_path: Path) -> bool:
    """
    Verify that the model can be loaded and has the expected structure
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        console.print(f"[blue]Verifying model at {model_path}...[/blue]")
        
        # Load the model
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        if not isinstance(checkpoint, dict):
            console.print("[red]❌ Model file is not in expected checkpoint format[/red]")
            return False
        
        if 'model_state_dict' not in checkpoint:
            console.print("[red]❌ Model file missing 'model_state_dict'[/red]")
            return False
        
        state_dict = checkpoint['model_state_dict']
        
        # Check for expected keys
        expected_keys = ['pred.0.weight', 'pred.0.bias', 'pred.2.weight', 'pred.2.bias']
        for key in expected_keys:
            if key not in state_dict:
                console.print(f"[red]❌ Model missing expected key: {key}[/red]")
                return False
        
        # Check dimensions
        first_layer_weight = state_dict['pred.0.weight']
        hidden_dim, input_dim = first_layer_weight.shape
        
        last_layer_weight = state_dict['pred.2.weight']
        output_dim, hidden_dim_check = last_layer_weight.shape
        
        if hidden_dim != hidden_dim_check:
            console.print(f"[red]❌ Hidden dimension mismatch: {hidden_dim} vs {hidden_dim_check}[/red]")
            return False
        
        if output_dim != 1:
            console.print(f"[red]❌ Output dimension should be 1, got {output_dim}[/red]")
            return False
        
        console.print(f"[green]✅ Model verification successful[/green]")
        console.print(f"[dim]   Architecture: {input_dim} -> {hidden_dim} -> {output_dim}[/dim]")
        
        if 'config' in checkpoint:
            config_info = checkpoint['config']
            if 'created_by' in config_info:
                console.print(f"[dim]   Created by: {config_info['created_by']}[/dim]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error verifying model: {e}[/red]")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download or create pre-trained MLP model")
    parser.add_argument(
        "--url", 
        type=str, 
        help="URL to download pre-trained model from"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force recreation even if model exists"
    )
    parser.add_argument(
        "--verify", 
        action="store_true", 
        help="Verify model after creation/download"
    )
    parser.add_argument(
        "--input-dim", 
        type=int, 
        default=4116, 
        help="Input dimension for placeholder model (default: 4116)"
    )
    parser.add_argument(
        "--hidden-dim", 
        type=int, 
        default=1024, 
        help="Hidden dimension for placeholder model (default: 1024)"
    )
    
    args = parser.parse_args()
    
    # Get model path from config
    model_path = Path(config.MLP_MODEL_PATH)
    
    console.print(f"[bold]SubgraphRAG+ MLP Model Setup[/bold]")
    console.print(f"Target path: {model_path}")
    
    # Check if model already exists
    if model_path.exists() and not args.force:
        console.print(f"[green]✅ Pre-trained MLP model already exists at {model_path}[/green]")
        if args.verify:
            verify_model(model_path)
        return 0
    
    success = False
    
    # Try to download from URL if provided
    if args.url:
        success = download_pretrained_model(args.url, model_path)
    
    # If download failed or no URL provided, create placeholder
    if not success:
        console.print("[yellow]Creating placeholder model for development/demo purposes...[/yellow]")
        success = create_placeholder_model(model_path, args.input_dim, args.hidden_dim)
        
        if success:
            console.print("\n[bold yellow]📝 Note for Production Use:[/bold yellow]")
            console.print("This is a placeholder model for development/demo purposes.")
            console.print("For production use, you should:")
            console.print("1. Train a real model using the SubgraphRAG training pipeline")
            console.print("2. Use a pre-trained model from the original SubgraphRAG repository")
            console.print("3. Download a model using: make get-pretrained-mlp --url <model_url>")
    
    # Verify the model if requested
    if success and args.verify:
        verify_model(model_path)
    
    if success:
        console.print(f"\n[green]🎉 MLP model setup complete![/green]")
        return 0
    else:
        console.print(f"\n[red]❌ MLP model setup failed![/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 