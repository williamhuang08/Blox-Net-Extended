from bloxnet.pybullet.place_blocks_in_json import init_pybullet
from tqdm.contrib.concurrent import process_map
import os
import dotenv
# from bloxnet.pipelines.parallel_whole_structure import generate_structure
from bloxnet.pipelines.whole_structure import generate_structure
from bloxnet.pybullet.place_blocks_in_json import init_pybullet
from bloxnet.utils.utils import load_from_json, write_error
from bloxnet.pipelines.ratings import get_ratings
from bloxnet.pipelines.bracket_selection import get_best_assembly
from perturbation_analysis.run_perturbation_analysis import run_perturbation_analysis

structure_name = "Bridge"

def make_pybullet(x):
    to_build, num_structures = x
    dotenv.load_dotenv()
    blockset = "blocksets/printed_blocks.json"
    blockset = load_from_json(blockset)

    init_pybullet(gui=False)

    try:
        assembly = generate_structure(to_build, blockset, iter=num_structures)
    except:
        return None

    # adds ratings to assembly
    get_ratings(assembly)

    return assembly

def main_run_pipeline_single_obj_parallel(to_build, num_structures=10):
    # Ensure this is only run in the main process
    assemblies = process_map(
        make_pybullet,
        zip(
            [to_build] * num_structures,
            range(num_structures)
        )
        )
    
    assemblies = list(filter(lambda x: x is not None, assemblies))

    init_pybullet(gui=False)
    # DO PERTURBATION ANALYSIS
    # assemblies = [run_perturbation_analysis(assembly) for assembly in assemblies]
    # ---------

    best_assembly = get_best_assembly(assemblies, use_stability=True)
    if best_assembly is None:
        write_error(f"{assemblies[0].structure_directory}/error.txt", "best_assembly is None, could mean all designs are unstable")

    # best_assembly = run_perturbation_analysis(best_assembly)

    # best_assembly.save_to_structure_dir(name = "best_assembly.pkl")
    if best_assembly is not None:
        best_assembly.human_friendly_save(f"{best_assembly.structure_directory}/best_assembly")


if __name__ == "__main__":
    main_run_pipeline_single_obj_parallel(structure_name)