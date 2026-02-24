import json
import os
from collections import defaultdict

def build_dag():
    with open('docs_metadata/import_evidence.json') as f:
        data = json.load(f)
    
    modules = data['modules']
    
    adj = defaultdict(list)
    nodes = set()
    
    # We only care about modules in thesis_pkg
    for mod_name in modules.keys():
        if mod_name.startswith('thesis_pkg'):
            nodes.add(mod_name)
            
    for mod_name in nodes:
        for imp in modules[mod_name].get('imports', []):
            if imp in nodes and imp != mod_name:
                adj[mod_name].append(imp)
                
    entry_points = [
        'thesis_pkg.pipelines.sec_pipeline',
        'thesis_pkg.pipelines.ccm_pipeline',
        'thesis_pkg.pipelines.sec_ccm_pipeline'
    ]
    
    visited = set()
    order = []
    
    def dfs(node, path):
        if node in visited:
            return
        if node in path:
            return # detect cycle and ignore back-edge
        path.add(node)
        
        # Sort dependencies for deterministic output
        for dep in sorted(adj[node]):
            dfs(dep, path)
            
        path.remove(node)
        visited.add(node)
        order.append(node)

    for ep in entry_points:
        if ep in nodes:
            dfs(ep, set())
            
    # Include any remaining thesis_pkg modules not reachable from the entry points
    for node in sorted(nodes):
        if node not in visited:
            dfs(node, set())
            
    # order is built bottom-up (deepest dependencies first).
    # Reversing it gives top-down ordering from the entry points.
    order.reverse()
    
    os.makedirs('docs/architecture', exist_ok=True)
    with open('docs/architecture/00_master_index.md', 'w') as f:
        f.write('# Master Execution Index\n\n')
        f.write('Topological sort of the execution graph anchored by pipeline entry points:\n\n')
        for i, mod in enumerate(order, 1):
            f.write(f"{i}. `{mod}`\n")
            
    print(f"Generated docs/architecture/00_master_index.md with {len(order)} modules.")
    
    # Write the order out as json so we can easily load it for Phase 2
    with open('docs_metadata/dag_order.json', 'w') as f:
        json.dump(order, f, indent=2)

if __name__ == "__main__":
    build_dag()
