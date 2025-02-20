import numpy as np
import pybullet as p


def get_block_stability(block, assembly):
    return assembly.structure.check_stability(block.id, place_all_blocks=False)[0]


def check_near_collision(block, assembly, collision_thresh=0.015):
    assembly.structure.place_blocks(drop=False)

    def _get_z_overlap(block_1, block_2):
        zdim_1, zdim_2 = block_1.dimensions[-1], block_2.dimensions[-1]
        z_pose_1, z_pose_2 = block_1.position[-1], block_2.position[-1]

        # Calculate the z-extents (min and max z-coordinates) for both blocks
        z_min_1 = z_pose_1 - zdim_1 / 2
        z_max_1 = z_pose_1 + zdim_1 / 2
        z_min_2 = z_pose_2 - zdim_2 / 2
        z_max_2 = z_pose_2 + zdim_2 / 2

        epsilon = 0.001
        # Check if the blocks overlap along the z-axis
        z_overlap = not (z_max_1 - epsilon <= z_min_2 or z_max_2 <= z_min_1 + epsilon)

        return z_overlap

    def _get_surface_distance(obj_id1, obj_id2):
        closest_points = p.getClosestPoints(
            obj_id1, obj_id2, distance=100
        )  # Get closest points with a large search radius
        if closest_points:
            return closest_points[0][
                8
            ]  # The distance between the two objects (surface-to-surface)
        print("WARNING: NO CLOSEST POINT FOUND")
        return None

    for other_block in assembly.structure.structure:
        if (
            other_block.id == block.id
        ):  # or block.id == 0 or other_block.id == 0: # I don't think structure would contain the ground plane ever?
            continue
        if (
            _get_z_overlap(block, other_block)
            and _get_surface_distance(block.id, other_block.id) < collision_thresh
        ):
            return True

    return False


def check_block_status(block, assembly, verbose=False):
    """
    Determines whether a block needs to be perturbed or not
    Returns True if in good standing, False if not
    """
    stable = get_block_stability(block, assembly)
    near_collision = check_near_collision(block, assembly)

    return stable and (not near_collision)


def circle_sample(radius, num_points, center):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.array(
        [[radius * np.cos(angle), radius * np.sin(angle), 0] for angle in angles]
    )
    points += np.array(center)
    return points


def needs_perturbation(block, assembly):
    def _near_instability(
        block,
        assembly,
        min_radius=1,
        max_radius=4,
        num_circles=2,
        num_points_per_circle=8,
    ):
        positions = []
        circle_radii = np.linspace(min_radius, max_radius, num_circles)
        original_coordinates = block.position

        for radius in circle_radii:
            positions.extend(
                circle_sample(
                    radius, num_points_per_circle, original_coordinates
                ).tolist()
            )

        for position in positions:
            block.position = position
            assembly.structure.set_block_by_id(block.id, block)
            # assembly.structure.place_blocks (drop=False)
            stable = get_block_stability(block, assembly)
            if not stable:
                print("stability based needs perturbation")
                return True
        return False

    return (not check_block_status(block, assembly)) or _near_instability(
        block, assembly
    )


if __name__ == "__main__":
    from bloxnet.structure import Assembly
    from bloxnet.pybullet.place_blocks_in_json import init_pybullet

    init_pybullet(gui=True)

    a = Assembly.load("airport.pkl")
    a.structure.place_blocks()
    for block in a.structure.structure:
        print(needs_perturbation(block, a))
