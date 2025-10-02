import tempfile
import pybullet as p
import pybullet_data
import time
import numpy as np

EPSILON = 1e-4


def init_pybullet(gui=False):
    # Connect to PyBullet if not already connected
    if not p.isConnected():
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

    # Set additional search path for PyBullet data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

def create_pyramid(name, base_dimensions, height, position, yaw_angle, color=[1, 0, 0, 1]):
    """
    Create a right pyramid (rectangular base) in PyBullet.

    Parameters:
    - name: str (unused, for symmetry)
    - base_dimensions: [L_mm, W_mm] base side lengths in millimeters
    - height: H_mm, apex height above base in millimeters
    - position: [x_mm, y_mm, z_mm] of the *centroid* (not base)
    - yaw_angle: degrees about +Z
    - color: [r,g,b,a]
    Notes:
    - For a uniform pyramid, the centroid is at H/4 above the base plane.
      So if you want the base at z = z0 (mm), pass position[2] = z0 + H/4.
    """

    # --- Units
    L_m = base_dimensions[0] / 1000.0
    W_m = base_dimensions[1] / 1000.0
    H_m = height             / 1000.0
    position_m = [x / 1000.0 for x in position]
    orientation = p.getQuaternionFromEuler([0.0, 0.0, np.radians(yaw_angle)])

    # --- Mass: V = (L*W*H)/3
    density = 1000.0  # kg/m^3
    volume  = (L_m * W_m * H_m) / 3.0
    mass    = density * volume

    # --- Vertices in local frame with origin at centroid
    # Base plane z = -H/4, apex z = +3H/4
    z_base = -H_m / 4.0
    z_apex = +3.0 * H_m / 4.0
    verts = np.array([
        [ +L_m/2.0, +W_m/2.0, z_base ],  
        [ +L_m/2.0, -W_m/2.0, z_base ],  
        [ -L_m/2.0, -W_m/2.0, z_base ],  
        [ -L_m/2.0, +W_m/2.0, z_base ],  
        [  0.0,       0.0,     z_apex ],
    ], dtype=np.float64)

    faces = [
        [0, 1, 2], [0, 2, 3], 
        [0, 1, 4], [1, 2, 4],
        [2, 3, 4], [3, 0, 4],
    ]

    col_id = p.createCollisionShape(
        shapeType=p.GEOM_CONVEX_HULL,
        vertices=verts.tolist()
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as f:
        obj_path = f.name
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    vis_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        rgbaColor=color
    )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=position_m,
        baseOrientation=orientation,
    )
    return body_id


def create_cuboid(name, dimensions, position, yaw_angle, color=[1, 0, 0, 1]):
    """
    Create a cuboid in a pybullet simulation environment.
    Parameters:
    - name: name of the block
    - dimensions: list of 3 integers, dimensions of the cuboid in mm
    - position: list of 3 integers, position of the cuboid in mm
    - yaw_angle: integer, yaw angle in degrees
    - color: list of 4 floats, RGBA color of the cuboid
    """
    dimensions_m = [x / 1000 for x in dimensions]
    position_m = [x / 1000 for x in position]
    orientation = p.getQuaternionFromEuler([0, 0, np.radians(yaw_angle)])
    half_extents = [dim / 2 for dim in dimensions_m]

    density = 1000  # Density of the block in kg/m^3
    mass = density * dimensions_m[0] * dimensions_m[1] * dimensions_m[2]

    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )
    id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position_m,
        baseOrientation=orientation,
    )

    return id


def create_cylinder(name, radius, height, position, color=[1, 0, 0, 1]):
    """
    Create a vertical cylinder in a pybullet simulation environment.
    Parameters:
    - name: name of the block
    - radius: integer, radius of the cylinder in mm
    - height: integer, height of the cylinder in mm
    - position: list of 3 integers, position of the cylinder in mm
    - color: list of 4 floats, RGBA color of the cylinder
    """
    orientation = [0, 0, 0, 1]
    position_m = [x / 1000 for x in position]
    radius_m = radius / 1000
    height_m = height / 1000

    density = 1000  # Density of the block in kg/m^3
    mass = density * np.pi * radius_m**2 * height_m

    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius_m, height=height_m
    )
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius_m, length=height_m, rgbaColor=color
    )
    id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position_m,
        baseOrientation=orientation,
    )

    return id


def check_body_collision(bodyID=-1):
    p.performCollisionDetection()
    contact_points = p.getContactPoints()

    for point in contact_points:
        bodyA = point[1]
        bodyB = point[2]

        if (bodyID == -1 or bodyID == bodyA or bodyID == bodyB) and point[8] < -EPSILON:
            return True

    return False


def move_until_contact(id, direction, step_size=0.01, debug=False):
    # normalize direction
    direction = np.array(direction) / np.linalg.norm(direction)

    # Keep moving the object until it makes contact with another object
    while not check_body_collision(id):
        # Get the current position and orientation of the object
        pos, ori = p.getBasePositionAndOrientation(id)
        new_pos = np.array(pos) + np.array(direction) * step_size
        p.resetBasePositionAndOrientation(id, new_pos, ori)
        if debug:
            time.sleep(0.002)


def move_out_of_contact(id):
    for _ in range(100):
        p.performCollisionDetection()
        contacts = p.getContactPoints()

        contact = None

        for c in contacts:
            if c[2] == id and c[8] <= -EPSILON:
                contact = c
                break
            elif c[1] == id and c[8] <= -EPSILON:
                contact = list(c)
                contact[1], contact[2] = contact[2], contact[1]
                contact[5], contact[6] = contact[6], contact[5]
                contact[7] = -np.array(contact[7])
                break

        if contact is None:
            return False

        # Extract relevant contact information
        bodyA = contact[1]
        bodyB = contact[2]
        posA = np.array(contact[5])
        posB = np.array(contact[6])
        normalB = np.array(contact[7])
        contactDistance = contact[8]

        # move body B in normalB * contactDistance
        bodyB_pos, bodyB_ori = p.getBasePositionAndOrientation(bodyB)
        p.resetBasePositionAndOrientation(
            bodyB, bodyB_pos + normalB * contactDistance, bodyB_ori
        )


def place_blocks(blocks, blockset):
    # Reset simulation
    p.resetSimulation()
    plane = p.loadURDF("plane.urdf")  # Load plane (ground)

    # set gravity
    p.setGravity(0, 0, -9.81)

    structure_json_w_ids = []

    # run AI code
    for block in blocks:

        # get block type

        block_type = blockset[block["name"]]["shape"]
        position = block["position"]
        yaw_angle = block["yaw_angle"]

        if block_type == "cuboid":
            dimensions = [
                blockset[block["name"]]["dimensions"]["x"],
                blockset[block["name"]]["dimensions"]["y"],
                blockset[block["name"]]["dimensions"]["z"],
            ]

            id = create_cuboid(
                block["name"], dimensions, position, yaw_angle, [1, 0, 0, 1]
            )

        elif block_type == "cylinder":
            radius = blockset[block["name"]]["dimensions"]["radius"]
            height = blockset[block["name"]]["dimensions"]["height"]

            id = create_cylinder(block["name"], radius, height, position, [1, 0, 0, 1])
        elif block_type == "pyramid":
            base_dimensions = [
                blockset[block["name"]]["dimensions"]["x"],
                blockset[block["name"]]["dimensions"]["y"],
            ]
            height = blockset[block["name"]]["dimensions"]["height"]
            id = create_pyramid(block["name"], base_dimensions, height, position, yaw_angle, [1, 0, 0, 1])

        # Move the block until it makes contact with another object
        move_until_contact(id, direction=[0, 0, -1], step_size=0.001, debug=True)

        # Move the block out of contact
        move_out_of_contact(id)

        position, orientation = p.getBasePositionAndOrientation(id)

        block_json = {
            "name": block["name"],
            "id": id,
            "position": position,
            "yaw_angle": yaw_angle,
        }
        structure_json_w_ids.append(block_json)

    return structure_json_w_ids


def move_block(block_num, answer, structure_json_w_ids):
    for block in structure_json_w_ids:
        if block["id"] == block_num:
            pos_x, pos_y, pos_z = block["position"]
            move_x, move_y, move_z = answer
            block["position"] = [pos_x + move_x, pos_y + move_y, pos_z + move_z]
            break
    return structure_json_w_ids


def rotate_block(block_num, answer, structure_json_w_ids):
    for block in structure_json_w_ids:
        if block["id"] == block_num:
            block["yaw"] += answer
            break
    return structure_json_w_ids


def delete_block(block_num, structure_json_w_ids):
    for i, block in enumerate(structure_json_w_ids):
        if block["id"] == block_num:
            del structure_json_w_ids[i]
            break
    return structure_json_w_ids
