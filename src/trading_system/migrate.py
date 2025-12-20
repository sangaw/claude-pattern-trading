import re

def extract_class(filename, class_name, output_file):
    """Extract a class from the monolithic file"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find class definition
    pattern = rf'class {class_name}.*?(?=\nclass |\n# ===|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        class_code = match.group(0)
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(f'"""\n{output_file}\n"""\n\n')
            f.write('import pandas as pd\n')
            f.write('import numpy as np\n')
            f.write('from config import TradingConfig\n\n')
            f.write(class_code)
        
        print(f"✓ Extracted {class_name} to {output_file}")
    else:
        print(f"✗ Could not find {class_name}")