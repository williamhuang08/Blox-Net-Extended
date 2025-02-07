"""
Script which goes through all objects in gpt_caching by loading best_assembly/assembly.pkl files and comparing their ratings. Returns top design names and their ratings.
"""
import os
from bloxnet.structure import Assembly

def get_sorted_assemblies_by_rating(gpt_cache_dir):
    assemblies = []
    for subdir in os.listdir(gpt_cache_dir):
        subdir_path = os.path.join(gpt_cache_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        assembly_path = os.path.join(subdir_path, 'best_assembly', 'assembly.pkl')
        
        if not os.path.exists(assembly_path):
            print(f"{assembly_path} doen't exist, continuing")
            continue

        assembly = Assembly.load(assembly_path)
        assemblies.append(assembly)
    
    sorted_assemblies = sorted(assemblies, key=lambda x: x.eval_rating, reverse=True)
    return sorted_assemblies

if __name__ == "__main__":
    sorted_assemblies = get_sorted_assemblies_by_rating("gpt_caching")
    for assembly in sorted_assemblies:
        print(assembly.to_build, assembly.eval_rating)