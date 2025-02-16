"""
visualize assembly with GUI in pybullet
"""
import pybullet as p
import time
from bloxnet.structure import Assembly
from bloxnet.pybullet.place_blocks_in_json import init_pybullet
from get_best_objects_by_ratings import get_sorted_assemblies_by_rating
import threading


def run_simulation(steps=10000):
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def visualize_assembly(assembly):
    init_pybullet(gui=True)
    assembly.structure.place_blocks()
    run_simulation(10000)


if __name__ == "__main__":
    ASSEMBLY_NAME = "box"
    visualize_assembly(
        Assembly.load(f"gpt_caching/{ASSEMBLY_NAME}/best_assembly/assembly.pkl")
    )
