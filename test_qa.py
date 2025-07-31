"""
Quick test script for local LLM Q&A generation
"""

import json
import requests
from pathlib import Path

def test_qa_generation():
    """Test Q&A generation with a simple example"""
    
    # Test content
    content = """# Getting Started with Godot

Godot is a free and open-source game engine. To create a new project:

1. Go to Project > New Project in the main menu
2. Choose between 2D and 3D projects
3. Set your project name and location

Example code:
```gdscript
extends Node

func _ready():
    print("Hello Godot!")
```

This simple script prints a message when the node is ready."""

    # Simplified prompt
    prompt = f'''Create exactly 3 Q&A pairs from this Godot documentation. Return only valid JSON array format:

Content: {content}

Format: [{{"question": "...", "answer": "...", "category": "...", "difficulty": "beginner"}}]

JSON array:'''

    # Call Ollama
    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "options": {
            "temperature": 0.3,
            "num_predict": 500
        },
        "stream": False
    }
    
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()["response"]
        print("Raw response:")
        print(result)
        print("\n" + "="*50 + "\n")
        
        # Try to parse JSON
        try:
            start_idx = result.find('[')
            end_idx = result.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = result[start_idx:end_idx]
                qa_pairs = json.loads(json_str)
                
                print("Parsed Q&A pairs:")
                for i, pair in enumerate(qa_pairs, 1):
                    print(f"\n{i}. Question: {pair['question']}")
                    print(f"   Answer: {pair['answer']}")
                    print(f"   Category: {pair['category']}")
                    print(f"   Difficulty: {pair['difficulty']}")
                    
                # Save to file
                output_file = Path("test_qa_output.json")
                with open(output_file, 'w') as f:
                    json.dump(qa_pairs, f, indent=2)
                    
                print(f"\n✅ Successfully generated {len(qa_pairs)} Q&A pairs")
                print(f"📁 Saved to: {output_file}")
                
            else:
                print("❌ No JSON array found in response")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
        
    else:
        print(f"❌ Ollama API error: {response.status_code}")

if __name__ == "__main__":
    test_qa_generation()
