"""
Extract all code cells from Jupyter notebook and create Python scripts
"""
import json
import os

def extract_notebook_code(notebook_path, output_path):
    """Extract all code from notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code_cells = []
    markdown_cells = []
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            if code.strip():  # Only add non-empty cells
                code_cells.append(code)
        elif cell['cell_type'] == 'markdown':
            markdown = ''.join(cell['source'])
            if markdown.strip():
                markdown_cells.append(markdown)
    
    # Write code to Python file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('OLA Driver Churn Analysis - Ensemble Learning\n')
        f.write('Extracted from Jupyter Notebook\n')
        f.write('"""\n\n')
        
        for i, code in enumerate(code_cells):
            f.write(f'\n# ===== Code Cell {i+1} =====\n')
            f.write(code)
            f.write('\n')
    
    # Write markdown to separate file
    markdown_path = output_path.replace('.py', '_markdown.txt')
    with open(markdown_path, 'w', encoding='utf-8') as f:
        for i, md in enumerate(markdown_cells):
            f.write(f'\n===== Markdown Cell {i+1} =====\n')
            f.write(md)
            f.write('\n\n')
    
    print(f"✅ Extracted {len(code_cells)} code cells to {output_path}")
    print(f"✅ Extracted {len(markdown_cells)} markdown cells to {markdown_path}")

if __name__ == "__main__":
    notebook_file = "OLA - Ensemble Learning .ipynb"
    output_file = "ola_analysis.py"
    
    extract_notebook_code(notebook_file, output_file)
