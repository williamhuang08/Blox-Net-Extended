import pybullet as p
import numpy as np
import json
from perturbation_analysis.transform_utils import within_pos_rot_threshold
from bloxnet.structure.block import Block
from bloxnet.structure.structure import (
    _create_cone,
    _create_cuboid,
    _create_cylinder,
    create_block,
)
import os

CONE_STL_PATH = os.path.join(os.path.dirname(__file__), "cone.stl")

EPSILON = 1e-4


def place_given_blocks(blocks):
    p.resetSimulation()
    plane = p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)

    for block in blocks:
        id = create_block(block)


def load_blocks(block_properties):
    new_block_properties = []

    for block_dict in block_properties:

        block = Block.from_json(json.dumps(block_dict))

        if block.shape == "cuboid":
            block_id = _create_cuboid(block)
        elif block.shape == "cylinder":
            block_id = _create_cylinder(block)
        elif block.shape == "cone":
            block_id = _create_cone(block)

        new_block_properties.append(block)

    return new_block_properties


def get_initial_positions(block_info):
    # create init pos dict
    init_pos = {}
    for block in block_info:
        id = block.id
        pos, ori = p.getBasePositionAndOrientation(id)
        init_pos[id] = {"position": pos, "orientation": ori}

    return init_pos


# Function to convert NumPy arrays to lists
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def set_block_position_and_orientation(block_id, new_position, new_orientation):
    p.resetBasePositionAndOrientation(block_id, new_position, new_orientation)
    p.resetBaseVelocity(block_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])


def reset_blocks(init_pos):
    for block_id in init_pos:
        initial_position = init_pos[block_id]["position"]
        initial_orientation = init_pos[block_id]["orientation"]
        set_block_position_and_orientation(
            block_id, initial_position, initial_orientation
        )


def check_body_collision(bodyID=-1):
    p.performCollisionDetection()
    contact_points = p.getContactPoints()

    for point in contact_points:
        bodyA = point[1]
        bodyB = point[2]

        if (bodyID == -1 or bodyID == bodyA or bodyID == bodyB) and point[
            8
        ] < -10 * EPSILON:
            return True

    return False


def get_num_collisions():
    p.performCollisionDetection()
    contact_points = p.getContactPoints()
    # filter out where contact distance is positive
    contact_points = [point for point in contact_points if point[8] < -10 * EPSILON]
    return len(contact_points)


def get_num_contact_points():
    p.performCollisionDetection()
    contact_points = p.getContactPoints()
    # filter out where contact distance is positive
    contact_points = [point for point in contact_points if point[8] <= EPSILON]
    return len(contact_points)


def check_for_collapse(
    id, start_pos, start_ori, position_threshold=0.1, rotation_threshold=0.1
):
    cur_pos, cur_ori = p.getBasePositionAndOrientation(id)
    return not within_pos_rot_threshold(
        np.array(start_pos),
        np.array(start_ori),
        np.array(cur_pos),
        np.array(cur_ori),
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
    )


def get_num_collapse(init_pos, position_threshold=0.1, rotation_threshold=0.1):
    num_collapses = 0
    for id in init_pos:
        start_pos = init_pos[id]["position"]
        start_ori = init_pos[id]["orientation"]
        if check_for_collapse(
            id, start_pos, start_ori, position_threshold, rotation_threshold
        ):
            num_collapses += 1
    return num_collapses


class Camera:
    def __init__(self):
        # pybullet cam
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.4, 0.4, 0.3],
            cameraTargetPosition=[0, 0, 0.1],
            cameraUpVector=[0, 0, 1],
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )

    def take_picture(self):
        # save image
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=500,
            height=500,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return rgbImg
