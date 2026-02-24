import json
import ast
import os
from pathlib import Path

def generate_phase2():
    with open('docs_metadata/dag_order.json') as f:
        modules = json.load(f)
        
    os.makedirs('docs/architecture', exist_ok=True)
    
    for idx, mod in enumerate(modules, 1):
        if mod == 'thesis_pkg':
            rel_path = '__init__.py'
        else:
            rel_path = mod.replace('thesis_pkg.', '').replace('.', '/')
            if os.path.isdir(f'src/thesis_pkg/{rel_path}'):
                rel_path = f"{rel_path}/__init__.py"
            else:
                rel_path = f"{rel_path}.py"
        
        src_path = Path('src/thesis_pkg') / rel_path
        md_file = f"docs/architecture/{idx:02d}_{mod.split('.')[-1]}.md"
        
        if not src_path.exists():
            continue
            
        with open(src_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        try:
            tree = ast.parse(source)
        except Exception as e:
            print(f"Failed to parse {src_path}: {e}")
            continue
            
        md = [f"# Module: `{mod}`", "", "## A. Signatures & Parameters", ""]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                md.append(f"### Class: `{node.name}`")
                md.append("")
            elif isinstance(node, ast.FunctionDef):
                md.append(f"### Function: `{node.name}`")
                md.append("")
                md.append("| Parameter Name | Type | Default Value | Code-Derived Purpose |")
                md.append("|---|---|---|---|")
                
                defaults = node.args.defaults
                args = node.args.args
                def_offset = len(args) - len(defaults)
                
                kw_defaults = node.args.kw_defaults
                kwonlyargs = node.args.kwonlyargs
                
                for i, arg in enumerate(args):
                    arg_name = arg.arg
                    arg_type = ast.unparse(arg.annotation) if getattr(arg, 'annotation', None) else "Any"
                    default_val = "Required"
                    if i >= def_offset:
                        default_val = ast.unparse(defaults[i - def_offset]) if hasattr(defaults[i - def_offset], 'value') or True else "..." 
                        try:
                            default_val = ast.unparse(defaults[i - def_offset])
                        except:
                            default_val = "..."
                    md.append(f"| `{arg_name}` | `{arg_type}` | `{default_val}` | {'' if default_val == 'Required' else '*(Optional)*'} |")
                
                for i, arg in enumerate(kwonlyargs):
                    arg_name = arg.arg
                    arg_type = ast.unparse(arg.annotation) if getattr(arg, 'annotation', None) else "Any"
                    d = kw_defaults[i]
                    default_val = "Required"
                    if d:
                        try:
                            default_val = ast.unparse(d)
                        except:
                            default_val = "..."
                    md.append(f"| `{arg_name}` | `{arg_type}` | `{default_val}` | {'' if default_val == 'Required' else '*(Optional, kw-only)*'} |")
                    
                md.append("")
                
        md.append("## B. Execution Logic & Transformations")
        md.append("")
        
        logic = []
        assumptions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ['join', 'group_by', 'with_columns', 'filter', 'select']:
                    # Extract the first argument if it's a string, to give more context
                    arg_context = ""
                    if getattr(node, 'args', None) and len(node.args) > 0 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        arg_context = f" on `{node.args[0].value}`"
                    logic.append(f"- Dataframe transformation: **{node.func.attr}**{arg_context}")
            elif getattr(node, 'value', None) is not None and isinstance(getattr(node, 'value', None), ast.Constant) or isinstance(node, ast.Constant):
                val = getattr(node, 'value', None)
                if val is None:
                    continue
                if isinstance(val, str):
                    if len(val) > 8 and ('\\' in val or '^' in val or '$' in val or '(?i)' in val):
                        assumptions.append(f"- **Regex Pattern**: `{val}`")
                    elif val in ['duplicate', 'success', 'error', 'no_match', 'missing', 'invalid', 'exact_match', 'fuzzy_match']:
                        assumptions.append(f"- **Routing/Status Code**: `{val}`")
                elif isinstance(val, int) and val in [30, 90, 180, 365, 5, 10]:
                    assumptions.append(f"- **Temporal/Threshold Assumption**: `{val}`")
                    
        if logic:
            md.extend(sorted(list(set(logic))))
        else:
            md.append("*No explicit DataFrame transformations detected.*")
        md.append("")
        
        md.append("## C. Hardcoded Assumptions & Boundary Conditions")
        md.append("")
        if assumptions:
            md.extend(sorted(list(set(assumptions))))
        else:
            md.append("*No obvious hardcoded assumptions detected.*")
            
        md.append("")
        with open(md_file, 'w', encoding='utf-8') as mf:
            mf.write("\n".join(md))
            
    print("Generated Phase 2 docs.")

if __name__ == "__main__":
    generate_phase2()
