import ast
import os
import re

def is_regex_call(node):
    """Check if an AST Call node is a regex-related function."""
    if isinstance(node.func, ast.Attribute):
        # e.g., re.compile, df.str.extract
        if isinstance(node.func.value, ast.Name) and node.func.value.id == 're':
            return True
        if node.func.attr in ('extract', 'contains', 'replace', 'findall', 'match', 'count') and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'str':
            return True
    return False

def extract_regexes_from_file(filepath):
    regexes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content, filename=filepath)
        
        for node in ast.walk(tree):
            # 1. Look for function calls to re.* or .str.*
            if isinstance(node, ast.Call) and is_regex_call(node):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    regexes.append({
                        'pattern': node.args[0].value,
                        'line': node.lineno,
                        'type': 'Function Call: ' + ast.unparse(node.func)
                    })
            
            # 2. Look for assignments to variables with 'pattern' or 'regex' in their name
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id.lower()
                        if 'pattern' in name or 'regex' in name or 're_' in name:
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                regexes.append({
                                    'pattern': node.value.value,
                                    'line': node.lineno,
                                    'type': f'Variable Assignment: {target.id}'
                                })
                                
            # 3. Look for explicit raw strings because they are highly indicative of regex in python (though not exclusively)
            # Actually, `ast` doesn't easily distinguish raw strings from normal strings after parsing, 
            # but we will rely on function calls and variable names mostly.

    except SyntaxError:
        print(f"Syntax error in {filepath}")
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    # Also fallback to a regex to find any raw string that looks like it's used as a pattern
    try:
        lines = content.splitlines()
        for i, line in enumerate(lines):
            # match r"..." or r'...' if the line contains re. or .str. or pattern
            if ('re.' in line or '.str.' in line or 'pattern' in line or 'regex' in line.lower()):
                raw_strings = re.findall(r'r[\'"](.*?)[\'"](?![\'"])', line)
                for rs in raw_strings:
                    # check if already captured
                    if not any(r['pattern'] == rs and r['line'] == i+1 for r in regexes):
                        regexes.append({
                            'pattern': rs,
                            'line': i+1,
                            'type': 'Raw String Context Match'
                        })
    except Exception:
        pass
        
    return regexes

def main():
    root_dir = r"c:\Users\erik9\Documents\SEC_Data\code\NLP_Thesis"
    output_dict = {}
    
    for subdir, _, files in os.walk(root_dir):
        if '.venv' in subdir or '__pycache__' in subdir or '.git' in subdir:
            continue
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(subdir, file)
                rel_path = os.path.relpath(filepath, root_dir)
                regexes = extract_regexes_from_file(filepath)
                if regexes:
                    output_dict[rel_path] = regexes

    with open("regex_documentation.md", "w", encoding="utf-8") as out:
        out.write("# Regex Expressions Used in the Project\n\n")
        out.write("This document lists all regular expressions identified in the codebase, grouped by file.\n\n")
        
        for file, regexes in sorted(output_dict.items()):
            out.write(f"## `{file}`\n\n")
            out.write("| Line | Pattern | Context | \n")
            out.write("|---|---|---|\n")
            
            for r in sorted(regexes, key=lambda x: x['line']):
                pattern_escaped = r['pattern'].replace('|', '\\|').replace('`', '\\`').replace('\n', '\\n')
                out.write(f"| {r['line']} | `{pattern_escaped}` | {r['type']} |\n")
            
            out.write("\n")

if __name__ == "__main__":
    main()
