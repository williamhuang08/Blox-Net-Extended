"""
For checking if designs use blocks which aren't part of the blockset or use too many blocks!
"""
from get_best_objects_by_ratings import get_sorted_assemblies_by_rating
import json

def verify_assembly(assembly, blockset_path):
    with open(blockset_path, "r") as f:
        blockset = json.load(f)
    
    block_dims_ls = [
        [block_info["number_available"], sorted(block_info["dimensions"].values())]
        for block_info in blockset.values()
    ]

    return_value = 0

    for block in assembly.structure.structure:
        dimensions_ls = [x[1] for x in block_dims_ls]
        if sorted(block.dimensions) in dimensions_ls:
            idx = dimensions_ls.index(sorted(block.dimensions))
            block_dims_ls[idx][0] -= 1
            if block_dims_ls[idx][0] < 0:
                # print(f"extra {dimensions_ls[idx]} used")
                return_value = 1
        else:
            # print("Block not in available:", block)
            return_value = 2
        
    return return_value


if __name__ == "__main__":
    assemblies = get_sorted_assemblies_by_rating("gpt_caching")

    verification_ls = [0] * len(assemblies)
    for i, a in enumerate(assemblies):
        verified = verify_assembly(a, "blocksets/printed_blocks.json")

        verification_ls[i] = verified == 1
        if verified == 1:
            print(a.to_build)
    
    print("total proportion: ", sum(verification_ls) / len(assemblies))

