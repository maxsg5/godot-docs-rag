#!/usr/bin/env python3
"""
Configuration Manager for Godot Documentation RAG Pipeline

This script allows easy switching between different document loading 
and text splitting configurations.
"""

import yaml
import argparse
from pathlib import Path
import sys


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the current configuration."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def save_config(config: dict, config_path: str = "config.yaml"):
    """Save configuration to file."""
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def show_current_config():
    """Display the current configuration."""
    try:
        config = load_config()
        print("📋 Current Configuration:")
        print("=" * 40)
        print(f"Document Loading Method: {config['document_loading']['method']}")
        print(f"Text Splitting Method: {config['text_splitting']['method']}")
        print(f"Secondary Splitting: {'enabled' if config['secondary_splitting']['enabled'] else 'disabled'}")
        print(f"Chunk Size: {config['secondary_splitting']['chunk_size']}")
        print(f"Chunk Overlap: {config['secondary_splitting']['chunk_overlap']}")
        print(f"Max Examples: {config['output']['max_examples_to_show']}")
        print("=" * 40)
    except FileNotFoundError:
        print("❌ Configuration file not found. Run with --reset to create default config.")


def update_config(args):
    """Update configuration based on command line arguments."""
    try:
        config = load_config()
    except FileNotFoundError:
        print("❌ Configuration file not found. Creating default config...")
        from SplitTextForRAG import create_default_config
        create_default_config("config.yaml")
        config = load_config()
    
    # Update document loading method
    if args.doc_method:
        if args.doc_method not in ['unstructured', 'bs4']:
            print("❌ Invalid document loading method. Use 'unstructured' or 'bs4'")
            return
        config['document_loading']['method'] = args.doc_method
        print(f"✅ Updated document loading method to: {args.doc_method}")
    
    # Update text splitting method
    if args.split_method:
        if args.split_method not in ['html_header', 'html_section', 'html_semantic']:
            print("❌ Invalid splitting method. Use 'html_header', 'html_section', or 'html_semantic'")
            return
        config['text_splitting']['method'] = args.split_method
        print(f"✅ Updated text splitting method to: {args.split_method}")
    
    # Update chunk size
    if args.chunk_size:
        config['secondary_splitting']['chunk_size'] = args.chunk_size
        print(f"✅ Updated chunk size to: {args.chunk_size}")
    
    # Update chunk overlap
    if args.chunk_overlap:
        config['secondary_splitting']['chunk_overlap'] = args.chunk_overlap
        print(f"✅ Updated chunk overlap to: {args.chunk_overlap}")
    
    # Toggle secondary splitting
    if args.toggle_secondary:
        current = config['secondary_splitting']['enabled']
        config['secondary_splitting']['enabled'] = not current
        print(f"✅ Secondary splitting {'enabled' if not current else 'disabled'}")
    
    # Update max examples
    if args.max_examples:
        config['output']['max_examples_to_show'] = args.max_examples
        print(f"✅ Updated max examples to: {args.max_examples}")
    
    save_config(config)
    print("💾 Configuration saved!")


def show_presets():
    """Show available configuration presets."""
    presets = {
        "fast": {
            "description": "Fast processing with HTML header splitting",
            "doc_method": "unstructured",
            "split_method": "html_header",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "semantic": {
            "description": "Semantic preservation with tables/lists intact",
            "doc_method": "unstructured", 
            "split_method": "html_semantic",
            "chunk_size": 1000,
            "chunk_overlap": 100
        },
        "detailed": {
            "description": "Detailed extraction with BeautifulSoup and sections",
            "doc_method": "bs4",
            "split_method": "html_section", 
            "chunk_size": 750,
            "chunk_overlap": 75
        }
    }
    
    print("🎛️ Available Configuration Presets:")
    print("=" * 50)
    for name, preset in presets.items():
        print(f"\n📦 {name.upper()}")
        print(f"   Description: {preset['description']}")
        print(f"   Document method: {preset['doc_method']}")
        print(f"   Split method: {preset['split_method']}")
        print(f"   Chunk size: {preset['chunk_size']}")
        print(f"   Chunk overlap: {preset['chunk_overlap']}")
    print("=" * 50)


def apply_preset(preset_name: str):
    """Apply a configuration preset."""
    presets = {
        "fast": {
            "doc_method": "unstructured",
            "split_method": "html_header",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "semantic": {
            "doc_method": "unstructured", 
            "split_method": "html_semantic",
            "chunk_size": 1000,
            "chunk_overlap": 100
        },
        "detailed": {
            "doc_method": "bs4",
            "split_method": "html_section", 
            "chunk_size": 750,
            "chunk_overlap": 75
        }
    }
    
    if preset_name not in presets:
        print(f"❌ Unknown preset: {preset_name}")
        print("Available presets: fast, semantic, detailed")
        return
    
    try:
        config = load_config()
    except FileNotFoundError:
        from SplitTextForRAG import create_default_config
        create_default_config("config.yaml")
        config = load_config()
    
    preset = presets[preset_name]
    config['document_loading']['method'] = preset['doc_method']
    config['text_splitting']['method'] = preset['split_method'] 
    config['secondary_splitting']['chunk_size'] = preset['chunk_size']
    config['secondary_splitting']['chunk_overlap'] = preset['chunk_overlap']
    
    save_config(config)
    print(f"✅ Applied '{preset_name}' preset configuration!")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Godot Documentation RAG Pipeline Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_config.py --show                    # Show current config
  python manage_config.py --doc-method bs4         # Switch to BeautifulSoup4
  python manage_config.py --split-method html_semantic  # Use semantic splitting
  python manage_config.py --chunk-size 2000        # Set chunk size to 2000
  python manage_config.py --preset semantic        # Apply semantic preset
  python manage_config.py --presets               # Show available presets
        """
    )
    
    # Information commands
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--presets', action='store_true', help='Show available presets')
    
    # Configuration updates
    parser.add_argument('--doc-method', choices=['unstructured', 'bs4'], 
                       help='Document loading method')
    parser.add_argument('--split-method', choices=['html_header', 'html_section', 'html_semantic'],
                       help='Text splitting method')
    parser.add_argument('--chunk-size', type=int, help='Secondary splitting chunk size')
    parser.add_argument('--chunk-overlap', type=int, help='Secondary splitting chunk overlap')
    parser.add_argument('--toggle-secondary', action='store_true', 
                       help='Toggle secondary splitting on/off')
    parser.add_argument('--max-examples', type=int, help='Max example documents to show')
    
    # Presets
    parser.add_argument('--preset', choices=['fast', 'semantic', 'detailed'],
                       help='Apply a configuration preset')
    
    # Reset
    parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.reset:
        from SplitTextForRAG import create_default_config
        create_default_config("config.yaml")
        print("✅ Configuration reset to defaults!")
        return
    
    if args.presets:
        show_presets()
        return
    
    if args.preset:
        apply_preset(args.preset)
        return
    
    if args.show or len(sys.argv) == 1:
        show_current_config()
        return
    
    # Update configuration
    update_config(args)


if __name__ == "__main__":
    main()
