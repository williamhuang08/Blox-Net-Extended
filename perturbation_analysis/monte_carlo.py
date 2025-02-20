import numpy as np
from perturbation_analysis.perturbation_utils import check_block_status
from bloxnet.structure.block import Block


def circle_sample(radius, num_points, original_coordinates):
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(t) + original_coordinates[0]
    y = radius * np.sin(t) + original_coordinates[1]
    return np.array(list(zip(x, y, np.repeat(original_coordinates[2], num_points))))


def analyze_monte_carlo(monte_carlo_results):
    x_successful_sum = 0
    y_successful_sum = 0

    x_failed_sum = 0
    y_failed_sum = 0

    num_successful = 0
    num_failed = 0

    for radius, values in monte_carlo_results.items():
        for i in range(len(values)):
            if values[i][1]:
                x_successful_sum += values[i][0][0]
                y_successful_sum += values[i][0][1]
                num_successful += 1
            else:
                x_failed_sum += values[i][0][0]
                y_failed_sum += values[i][0][1]
                num_failed += 1
    if num_successful == 0:
        x_mean = None
        y_mean = None
    else:
        x_mean = x_successful_sum / num_successful
        y_mean = y_successful_sum / num_successful

    return x_mean, y_mean


def run_monte_carlo(
    block: Block,
    assembly,
    num_circles=10,
    min_radius=0.5,
    max_radius=5,
    original_position_status=False,
    debug=False,
    pos_threshold=0.1,
    rot_threshold=0.1,
    num_points_per_circle=8,
):
    monte_carlo_positions = {}
    circle_radii = np.linspace(min_radius, max_radius, num_circles)
    original_coordinates = block.position
    monte_carlo_results = {}

    for radius in circle_radii:
        monte_carlo_positions[radius] = circle_sample(
            radius, num_points_per_circle, original_coordinates
        )
        monte_carlo_results[radius] = []

    for radius, positions in monte_carlo_positions.items():
        for i, position in enumerate(positions):
            block.position = position
            assembly.structure.set_block_by_id(block.id, block)
            assembly.structure.place_blocks()
            verbose = False
            block_status = check_block_status(block, assembly, verbose)

            monte_carlo_results[radius].append([position, block_status])

    monte_carlo_results[0] = [[original_coordinates, original_position_status]]

    new_x, new_y = analyze_monte_carlo(monte_carlo_results)
    if not (new_x == None):
        block.position = [new_x, new_y, original_coordinates[2]]
    else:
        block.position = original_coordinates

    if debug:
        print(assembly.structure.get_block_by_id(block.id).position)
        print("OLD BLOCK POSES", original_coordinates)
        print("NEW BLOCK POSES", new_x, new_y)

    return assembly
