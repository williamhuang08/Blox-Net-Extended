import json
import os
import numpy as np
import pybullet as p

import matplotlib.pyplot as plt
from PIL import Image

from bloxnet.utils.utils import (
    get_last_json_as_dict,
    save_to_json,
    slugify,
    markdown_json,
)
from bloxnet.pybullet.pybullet_images import get_imgs
from bloxnet.prompts.prompt import prompt_with_caching
from bloxnet.structure import Structure, Block, Assembly

SAVE_DIR = "gpt_caching"


def make_description_prompt(to_build):
    return f"""
I'm working on constructing a block tower that represents a(n) {to_build}. I need a concise, qualitative description of the design that captures its essence in a minimalistic style. The design should focus on simplicity, avoiding unnecessary complexity while still conveying the key features of a(n) {to_build}. The description should highlight the overall structure and proportions, emphasizing how the block arrangement reflects the object's shape and form. However the design shouldn't be too large, too wide, or too tall.
""".strip()


def make_plan_prompt(to_build, blockset, description):
    return [
        f"""
Here's a description of the layout of a {to_build}:
{description}

You have the following blocks available: 
{markdown_json(blockset)}
Write a plan for how to assemble a {to_build} using the available blocks. Use blocks as needed while respecting the number available constraint. 

Explain which blocks to use and their shape and dimensions. 

Explain the overall orientation of the structure.

Explain each block's role in the structure. 

Explain how the blocks should stack on top of each other (they can also just be placed on the ground). 

Do not overcomplicate the design. Try to use a minimal number of blocks to represent the key components of a {to_build}. Avoid making structures that are too tall, wide, or complex.

Only consider the main key components of a {to_build}, not minor details that are hard to represent with blocks. 
Use a minimal amount of blocks and keep it simple, just enough so that it looks like a {to_build}.

The dimensions of a cuboid are given as x, y, and z, which define the size of the block. You can rearrange these dimensions to fit your design requirements. For instance, if you need to place a block "vertically" with its longest side along the z-axis, but the dimensions are listed as x: 30, y: 30, z: 10, you can adjust them to x: 10, y: 30, z: 30 to achieve the desired orientation. Ensure the x and y dimensions are also consistent with the rest of the design.

Cylinders are always positioned "upright," with the flat sides parallel to the ground and their radius extending along the x and y axes.

Cones are always positioned with their flat side down and their pointed tip facing upwards. This means the base of the cone lies parallel to the ground plane, with the cone's height extending along the z-axis and the radius along the x and y axes.

Pyramids have a flat base and a pointed apex. They should be placed with the base down and apex up for stability unless specifically used as caps. Avoid placing other blocks directly on the apex; place on the flat base or along large faces with enough support.

Decide a semantic name for the block for the role it represents in the structure. 
Decide the colors of each block to look like a {to_build}. Color is an rgba array with values from 0 to 1.
"""
    ]


def order_blocks_prompts(to_build):
    return f"""
Given the blocks described in the plan, I will place and stack these blocks one at a time by lowering them from a very tall height.

Please describe the sequence in which the blocks should be placed to correctly form a {to_build} structure. This means that blocks at the bottom should be placed first, followed by the higher blocks, so that the upper blocks can be stacked on top of the lower ones. Also note that it is difficult to stack blocks on top of a cone, so avoid placing blocks directly on top of cones. Also avoid placing blocks directly on a pyramid's apex; if stacking on a pyramid, use a flat cap (e.g., a small cuboid) or place along a sufficiently supported, gentle face.


For each block, specify whether it will be placed on the ground or on top of another block. If a block will be supported by multiple blocks, mention all of them. Ensure that the blocks are placed in a way that they remain stable and won't topple over when the physics simulation is run. Blocks cannot hover without support.
""".strip()


def decide_positions_prompts(to_build):
    return f"""
With the stacking order determined, I now need to know the x and y positions, as well as the yaw angle (in degrees), for each block to build a {to_build} structure.

The x and y coordinates should represent the center of each block. The yaw angle refers to the rotation around the z-axis in degrees. Remember, you can swap the dimensions of blocks to adjust their configuration.

Ensure that blocks at similar heights in the structure are spaced out in x and y so that they don't collide.

Make sure the structure is roughly centered at the origin (0, 0), and that each block stacks correctly on the specified blocks (or the ground). Every block must have a stable base to prevent it from falling. 

Consider the dimensions of the blocks when determining the x, y positions. Provide your reasoning for the chosen x and y positions and the yaw angle for each block.

Output a JSON following this format:
{markdown_json(
    [
        {
            "name": "support1",
            "shape": "cylinder",
            "dimensions": {"radius": 20, "height": 40},
            "color": [0.5, 0.5, 0.5, 1],
            "position": {"x": -50, "y": 0},
            "yaw": 0,
        },
        {
            "name": "support2",
            "shape": "cylinder",
            "dimensions": {"radius": 20, "height": 40},
            "color": [0.5, 0.5, 0.5, 1],
            "position": {"x": 50, "y": 0},
            "yaw": 0,
        },
        {
            "name": "deck",
            "shape": "cuboid",
            "dimensions": {"x": 100, "y": 50, "z": 20},
            "color": [0.5, 0.5, 0.5, 1],
            "position": {"x": 0, "y": 0},
            "yaw": 45,
        },
        {
            "name": "roof",
            "shape": "pyramid",
            "dimensions": {"base": 100, "height": 50},
            "color": [0.5, 0.5, 0.5, 1],
            "position": {"x": 0, "y": 0},
            "yaw": 45,
        },
    ]
)}
"""


def get_stability_correction(
    to_build, unstable_block: Block, pos_delta, structure_json, x_img, y_img
):
    return [
        f"""
{markdown_json(structure_json)}

While building the {to_build} by placing blocks one at a time in the order you specified by the JSON above, I noticed that block {unstable_block.gpt_name} is unstable and falls. 
The block moved by {pos_delta[0]:.2f} mm in the x direction and {pos_delta[1]:.2f} mm in the y direction.
Please adjust the position of block {unstable_block.gpt_name} (And potentially other blocks) to make the structure more stable.
Make sure every block has a stable base to rest on.

Output the JSON with your corrections following the same format and provide some reasoning for the changes you made. Feel free to correct other parts of the structure if they appear incorrect or to add, change, or remove blocks.

Here is an orthographic image of the side view of the structure with the y-axis pointing to the right and the z-axis pointing up. {unstable_block.gpt_name} is highlighted in red while the other blocks are colored in white.
""",
        x_img,
        f"""
Here is an orthographic image of the side view of the structure with the x-axis pointing to the right and the z-axis pointing up. {unstable_block.gpt_name} is highlighted in red while the other blocks are colored in white.
""",
        y_img,
        f"""
Describe what you see in these images and use them to help inform your correction. Then, provide the ouptut JSON in the proper format.
""",
    ]


def get_structure_info(img):
    return [
        img,
        f"""
I am currently building a structure made out of toy blocks shown in the given image. Describe in detail as much as you can about this image. Please list the top 10 things that the structure most resembles in order of similarity.

After describing the image in detail and providing some initial thoughts, answer as JSON in the following format providing 10 guesses. Your guesses should mostly be single words like "bottle" and never use adjectives like "toy_bottle". 
{
    markdown_json({"guesses": ["guess_1", "guess_2", "guess_3", "guess_4", "guess_5", "guess_6", "guess_7", "guess_8", "guess_9", "guess_10"]})
}
""".strip(),
    ]


def get_structure_rating(to_build):
    return f"""
Given your description of the block structure, how well does the structure in the image use blocks to resemble a {to_build} considering that the structure is made from  a limited set of toy blocks? Rate the resemblance of the block structure to a {to_build} on a scale of 1 to 5 defined by the following:
1 - the structure in the image has no resemblance to the intended structure. It's missing all key features and appears incoherent
2 - the structure in the image has a small amount of resemblance to the intented structure. It has at least 1 key feature and shows an attempt at the intended structure
3 - the structure has clear similarities to the intended structure and appears coherent. It has at least 1 key feature and shows similarities in other ways as well.
4 - the structure represents multiple key features of the intended design and shows a decent overall resemblance.
5 - the structure in the image is a good block reconstruction of the intended structure, representing multiple key features and showing an overall resemblance to the intended structure.

Provide a brief explanation of your though process then provide your final response as JSON in the following format:
{markdown_json({"rating": 5})}
"""


def process_available_blocks(blocks):
    available_blocks = []
    for block_name, block in blocks.items():
        block_shape = block["shape"]
        block_dimensions = block["dimensions"]
        number_available = block["number_available"]
        available_blocks.append(
            {
                "shape": block_shape,
                "dimensions": block_dimensions,
                "number_available": number_available,
            }
        )
    return available_blocks


def blocks_from_json(json_data):
    blocks = []
    for block_data in json_data:
        if block_data["shape"] == "cuboid":
            dimensions = [
                block_data["dimensions"]["x"],
                block_data["dimensions"]["y"],
                block_data["dimensions"]["z"],
            ]
        elif block_data["shape"] == "cylinder" or block_data["shape"] == "cone":
            dimensions = [
                block_data["dimensions"]["radius"],
                block_data["dimensions"]["height"],
            ]
        elif block_data["shape"] == "pyramid":
            # Accept either square or rectangular base
            if "base" in block_data["dimensions"] and "height" in block_data["dimensions"]:
                base = block_data["dimensions"]["base"]
                height = block_data["dimensions"]["height"]
                dimensions = [base, base, height]          # [L, W, H]
            else:
                raise ValueError(
                    "pyramid dimensions must be {'base', 'height'} "
                    "or {'length','width','height'} (mm)"
                )
        else:
            raise ValueError(f"Invalid shape {block_data['shape']}")

        block = Block(
            id=999,  # id gets updated by place blocks call, otherwise it's unknown
            gpt_name=block_data["name"],
            block_name="",
            shape=block_data["shape"],
            dimensions=dimensions,
            position=[
                block_data["position"]["x"],
                block_data["position"]["y"],
                1 * 1000,
            ],
            orientation=p.getQuaternionFromEuler([0, 0, np.radians(block_data["yaw"])]),
            color=block_data["color"],
        )
        blocks.append(block)

    return blocks


def stability_check(blocks, debug=False):
    for i in range(len(blocks)):
        structure = Structure()
        structure.add_blocks(blocks[: i + 1])
        structure.place_blocks()

        last_block = blocks[i]

        x_img, y_img = get_imgs(
            keys=["x", "y"], axes=False, labels=False, highlight_id=last_block.id
        )

        stable, pos_delta, rot_delta = structure.check_stability(
            blocks[i].id, debug=debug
        )

        pos_delta = 1000 * np.array(pos_delta)

        if not stable:
            return False, last_block, pos_delta, x_img, y_img

    return True, None, None, x_img, y_img


def get_system_message():
    return f"""
You are an expert in creating block constructions with experience building many objects and creating stable structures.

Cuboid dimensions x, y, z define the size of the block, but you can swap them around to adjust the block's orientation. For example, if a block needs to be placed "vertically" then it's longest axis should be the z-axis. So, if a block is listed as having dimensions {{x: 50, y: 10, z: 10}} you can instead write the dimensions as {{x: 10, y: 10, z: 50}} listing z as the longest axis. It can also be important to pay attention to switching x and y to conform with the rest of your structure.

Cylinders are always upright with the z-axis as the height and the radius is the same along the x and y axes. 

Pyramids are placed with their flat base on the ground (or on another flat surface) and the apex pointing upward. Do not place blocks on the apex tip; use a flat cap or rest on broad faces with adequate support.

Don't make structures which are too complex or try to show fine detail. Instead, focus on showing the broader structure.
""".strip()


TEMPERATURE = 0.5


def generate_structure(to_build, available_blocks, iter=0):
    # make directory to save structure
    to_build_slug = slugify(to_build)
    structure_dir = os.path.join(SAVE_DIR, to_build_slug)
    os.makedirs(structure_dir, exist_ok=True)

    # prepare blockset
    available_blocks = process_available_blocks(available_blocks)

    # make description
    prompt = make_description_prompt(to_build)
    response, _ = prompt_with_caching(
        prompt,
        [],
        structure_dir,
        "description",
        cache=True,
        temeprature=TEMPERATURE,
        i=iter,
    )
    print(response)

    # # make plan
    prompt = make_plan_prompt(to_build, available_blocks, response)
    response, main_context = prompt_with_caching(
        prompt,
        [],
        structure_dir,
        "main_plan",
        cache=True,
        temeprature=TEMPERATURE,
        i=iter,
    )
    print(response)

    # decide ordering of blocks
    prompt = order_blocks_prompts(to_build)
    response, main_context = prompt_with_caching(
        prompt,
        main_context,
        structure_dir,
        "order_plan",
        cache=True,
        temeprature=TEMPERATURE,
        i=iter,
    )
    print(response)

    # decide positions
    prompt = decide_positions_prompts(to_build)
    response, main_context = prompt_with_caching(
        prompt,
        main_context,
        structure_dir,
        "positions_plan",
        cache=True,
        temeprature=TEMPERATURE,
        i=iter,
    )
    print(response)

    json_output = get_last_json_as_dict(response)
    blocks = blocks_from_json(json_output)

    structure = Structure()
    structure.add_blocks(blocks)
    structure.place_blocks()
    isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
    img = Image.fromarray(isometric_img)
    img.save(f"{structure_dir}/{to_build_slug}_{iter}.png")

    save_to_json(structure.get_json(), f"{structure_dir}/{to_build_slug}.json")

    for i in range(2):

        stable, unstable_block, pos_delta, x_img, y_img = stability_check(
            blocks, debug=True
        )

        if stable:
            break

        prompt = get_stability_correction(
            to_build, unstable_block, pos_delta, json_output, x_img, y_img
        )
        response, stability_context = prompt_with_caching(
            prompt, [], structure_dir, f"stability_correction_{iter}", cache=True, i=i
        )

        json_output = get_last_json_as_dict(response)
        blocks = blocks_from_json(json_output)

        structure = Structure()
        structure.add_blocks(blocks)
        structure.place_blocks()
        isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
        img = Image.fromarray(isometric_img)
        img.save(
            f"{structure_dir}/{to_build_slug}_stability_correction_{iter}_{i}.png"
        )

    assembly = Assembly(
        structure=structure,
        structure_directory=structure_dir,
        to_build=to_build,
        isometric_image=img,
        available_blocks_json=available_blocks,
        assembly_num=iter,
        eval_rating=None,
        eval_guesses=None,
    )
    assembly.save_to_structure_dir()
    return assembly
