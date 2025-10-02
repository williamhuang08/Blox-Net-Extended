import numpy as np
import pybullet as p

def _combined_friction(mu_a, mu_b, rule="min"):
    if rule == "min":
        return min(mu_a, mu_b)
    elif rule == "geom":
        return float(np.sqrt(mu_a * mu_b))
    else:  # conservative default
        return min(mu_a, mu_b)

def _body_friction(body_id):
    try:
        di = p.getDynamicsInfo(body_id, -1)
        # tuple: (mass, lateralFriction, rollingFriction, spinningFriction, ...
        mu = di[1] if di and di[1] is not None else 0.5
        return float(mu)
    except Exception:
        return 0.5

def will_slide_off_pyramid_face(block, assembly, nz_base_thr=0.985, rule="min", margin_eps=0.0):
    """
    Returns (True/False, details) — True if any contact with a pyramid face is predicted to slip.
    - nz_base_thr ~ 0.985 marks ~10° tilt as 'base-like'; above this we treat as flat base.
    - rule: 'min' (conservative) or 'geom' (geometric mean) for friction combine.
    - margin_eps: require mu_eff - tan(theta) >= margin_eps to consider 'safe'.
    """
    # Ensure bodies are placed and contacts are up-to-date
    assembly.structure.place_blocks(drop=False)

    this_id = block.id
    if this_id is None or this_id < 0:
        return False, {}

    # friction for this block
    mu_a = _body_friction(this_id)

    # Scan contacts with all other bodies
    for other in assembly.structure.structure:
        if other.id == this_id:
            continue
        if getattr(other, 'shape', None) != 'pyramid':
            continue

        mu_b = _body_friction(other.id)
        mu_eff = _combined_friction(mu_a, mu_b, rule=rule)

        # Check all contact points between (block, pyramid)
        cps = p.getContactPoints(this_id, other.id)
        for cp in cps:
            # In many pybullet builds:
            # cp[7] = contactNormalOnB (world), pointing from B towards A
            # cp[8] = contact distance (<=0 at contact)
            try:
                nBx, nBy, nBz = cp[7]
            except Exception:
                # Fallback: some builds store it differently; skip if missing
                continue

            # If |nBz| ~ 1 -> normal nearly vertical => plane nearly horizontal => base contact (not slipping case)
            nz = abs(float(nBz))
            if nz >= nz_base_thr:
                # near-horizontal face: treat as base; no slope-induced sliding
                continue

            # Slope angle theta (plane tilt from horizontal) equals angle between normal and +Z
            # alpha = arccos(nz) in [0, pi/2]; slope theta = alpha
            theta = float(np.arccos(np.clip(nz, -1.0, 1.0)))

            # No-slip condition on an incline: mu_eff >= tan(theta)
            slip_margin = mu_eff - np.tan(theta)
            if slip_margin < margin_eps:
                details = {
                    "other_pyramid_id": other.id,
                    "mu_eff": mu_eff,
                    "theta_deg": np.degrees(theta),
                    "tan_theta": np.tan(theta),
                    "slip_margin": slip_margin,
                    "contact_normal_world_onB": (nBx, nBy, nBz),
                }
                return True, details

    return False, {}



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

    will_slide, slip_info = will_slide_off_pyramid_face(block, assembly, nz_base_thr=0.985, rule="min", margin_eps=0.0)

    if verbose and will_slide:
        print("[SLIP WARNING]", slip_info)

    return stable and (not near_collision) and (not will_slide)


def circle_sample(radius, num_points, center):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.array(
        [[radius * np.cos(angle), radius * np.sin(angle), 0] for angle in angles]
    )
    points += np.array(center)
    return points


def needs_perturbation(block, assembly):
    slipping, _ = will_slide_off_pyramid_face(block, assembly)
    if slipping:
        return True
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
