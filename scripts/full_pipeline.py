"""
pipeline for running single object parallel on all target objects
"""
from tqdm.contrib.concurrent import process_map
from scripts.run_pipeline_single_obj_parallel import main_run_pipeline_single_obj_parallel
from bloxnet.utils.utils import write_error, slugify
import os

structure_names = ['Giraffe', 'Path', 'Spaceship', 'Earthquake', 'Fountain', 'Roman Column', 'Stairs', 'UFO', 'Bench', 'Cross', 'Island', 'Circus', 'Cow', 'Billboard', 'Rook Chess Piece', 'Train Station', 'Queen Chess Piece', 'Key', 'Universe', 'Train', 'Cliff', 'Cinema', 'Aquarium', 'Bird', 'Asteroid', 'Ant', 'Pig', 'Playground', 'Pool', 'Ramp', 'Tower', 'Rainbow', 'Alley', 'Medal Podium', 'Motorcycle', 'Palace', 'Square', 'Hotel', 'Bishop Chess Piece', 'Star', 'Box', 'Burger', 'Satellite', 'Lightening', 'Turtle', 'Door', 'Statue', 'Fence', 'Heart', 'Ferris Wheel', 'Sun', 'Waterfall', 'Gate', 'Column', 'Airport', 'Dog', 'Plane', 'Monument', 'Submarine', 'Barrier', 'Rain', 'Library', 'Lighthouse', 'Window', 'Igloo', 'Truck', 'Mountain', 'Windmill', 'Spider', 'Knight Chess Piece', 'Pedestal', 'Factory', 'Cabin', 'Archway', 'Cloud', 'Forest', 'Duck', 'Hut', 'Chair', 'Ladder', 'Road', 'Theater', 'Tent', 'Underpass', 'Beach', 'Street', 'Swing', 'Banner', 'Bee', 'Pyramid', 'Stool', 'Water Tower', 'Lollipop', 'Tractor', 'Table', 'Horse', 'TV', 'Roller Coaster', 'Tunnel', 'Car', 'Sheep', 'Swan', 'House', 'Whale', 'Astronaut', 'Volcano', 'Jungle', 'Sandbox', 'Flag', 'Traffic Light', 'Top Hat', 'Boat', 'Temple', 'Dinosaur', 'Elephant', 'Pillar', 'Eruption', 'Lamp', 'Museum', 'Alien', 'Arrow', 'Wall', 'Street Lamp', 'Barricade', 'Space Station', 'Cave', 'Clock', 'Gazebo', 'Sign', 'Crosswalk', 'Stadium', 'Greek Column', 'Park', 'Bridge', 'Castle', 'Slide', 'Flower', 'Helicopter', 'Robot', 'Tree', 'Ladybug', 'Rocket', 'Pawn Chess Piece', 'Obelisk', 'Arch', 'Highway', 'Overpass', 'Shark', 'Butterfly', 'Orchard', 'King Chess Piece', 'Bicycle', 'Kite', 'Cat', 'Store', "Empire State Building", "Parthenon", "Eifel Tower", "Taj Mahal", "Ceiling Fan", "Soccer Goal", "Tote Bag", "Well", "Teddy Bear", "Computer Monitor", "Coaster", "Closed Box", "Flower Pot", "Shelf", "Filament Roll", "Screw", "Couch", "Sofa", "Calipers", "Glasses", "Letter A", "Letter B", "Letter C", "Letter D", "Letter E", "Letter F", "Letter G", "Letter H", "Letter I", "Letter J", "Letter K", "Letter L", "Letter M", "Letter N", "Letter O", "Letter P", "Letter Q", "Letter R", "Letter S", "Letter T", "Letter U", "Letter V", "Letter W", "Letter X", "Letter Y", "Letter Z"]

def error_catch_main_run_pipeline_single_obj_parallel(to_build):
    try:
        main_run_pipeline_single_obj_parallel(to_build)
    except Exception as e:
        print(f"{to_build} errored")
        print(e)
        os.makedirs("errors", exist_ok=True)
        write_error(f"errors/error_{slugify(to_build)}", f"{e}")

if __name__ == "__main__":
    import time
    start = time.time()
    print(start)
    process_map(error_catch_main_run_pipeline_single_obj_parallel, structure_names)
    print(time.time() - start)