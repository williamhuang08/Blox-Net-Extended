from bloxnet.utils.utils import get_last_json_as_dict, slugify, markdown_json
from bloxnet.prompts.prompt import prompt_with_caching
from bloxnet.structure import Assembly
from typing import List
from bloxnet.pipelines.whole_structure import stability_check
import os
import numpy as np
from PIL import Image


def make_selection_prompt(to_build, img1, img2):
    return [
        f"""
You are an excellent critic of block assemblies and designs with helpful feedback and judgements.

Describe the following image (image 1) which is meant to represent a {to_build} made from blocks. In your explanation, consider which key features of the intended object are represented and whether the overall structure appears coherent:
""".strip(),
        img1,
        f"""
Then, describe this second image (image 2) which represents a different version of a {to_build} made from blocks. Again, consider key features and the overall strcture in your explanation:
""",
        img2,
        f"""
Based on the images and your descriptions, does image 1 or image 2 better resemble the desired structure? Answer in the following format where an answer of 0 indicates the first image better represents the target structure and an answer of 1 indicates the second image better represents the target structure:
{markdown_json({"answer": 0})}
or
{markdown_json({"answer": 1})}
""".strip(),
    ]


def load_images_from_directory(directory_path):
    image_arrays = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            file_path = os.path.join(directory_path, filename)

            with Image.open(file_path) as img:
                img_array = np.array(img)
                image_arrays.append(img_array)

    return image_arrays


def select(to_build, image_ls, structure_dir, cur_iteration=0):
    """
    takes in intended structure, list of images to select from
    comparison_prompt is a function which takes to_build, image 1, image 2 and returns 1 or 2
    cur_iteration is optionally for saving in prompt_w_cache
    returns index, image which is selected.
    """

    def recursive_compare(remaining_images, compare_function):
        """
        compare_function should return 0 for img1 and 1 for img2
        """
        num_images = len(remaining_images)

        # One image left -> winner
        if num_images == 1:
            return remaining_images[0]

        if num_images % 2 == 1:
            remaining_images.append((None, -1))

        next_round = []
        for i in range(0, len(remaining_images), 2):
            img1, idx1 = remaining_images[i]
            img2, idx2 = remaining_images[i + 1]

            # If there's no second image (bye situation), move the first image to the next round
            if img2 is None:
                winner = (img1, idx1)
            else:
                winner = compare_function(img1, img2)
                print(winner)
                # Preserve the index of the winning image
                winner = (
                    (image_ls[idx1], idx1) if winner == 0 else (image_ls[idx2], idx2)
                )

            next_round.append(winner)

        return recursive_compare(next_round, compare_function)

    compare_function_calls = 0

    def compare_function(to_build, img1, img2):
        nonlocal compare_function_calls
        compare_function_calls += 1

        to_build_slug = slugify(to_build)
        prompt = make_selection_prompt(to_build, img1, img2)

        response, main_context = prompt_with_caching(
            prompt,
            [],
            structure_dir,
            f"compare_{compare_function_calls}_" + to_build_slug,
            cache=True,
            i=cur_iteration,
        )

        json_output = get_last_json_as_dict(response)

        return json_output["answer"]

    image_list_with_indices = [(img, idx) for idx, img in enumerate(image_ls)]

    final_img, idx = recursive_compare(
        image_list_with_indices,
        compare_function=lambda x, y: compare_function(to_build, x, y),
    )
    return idx, final_img


def get_best_assembly(assemblies: List[Assembly], use_rating=True, use_stability=False):

    # Assemblies may still be unstable if after two iterations they weren't fixed!
    if use_stability:
        filtered_assemblies = [
            assembly
            for assembly in assemblies
            if stability_check(assembly.structure.structure)[0]
        ]

        if len(filtered_assemblies) == 0:
            print("-------------------------------------")
            print("WARNING: No stable assembly generated")
            print("Returning best assembly ignoring stability")
            print("-------------------------------------")
        else:
            assemblies = filtered_assemblies

    max_rating = max([assembly.eval_rating for assembly in assemblies])

    if use_rating:
        top_assemblies = list(filter(lambda x: x.eval_rating == max_rating, assemblies))
    else:
        top_assemblies = assemblies

    if len(top_assemblies) == 0:
        print("no stable assembly found")
        return None

    if len(top_assemblies) == 1:
        return top_assemblies[0]

    to_build, structure_directory = (
        top_assemblies[0].to_build,
        top_assemblies[0].structure_directory,
    )

    image_ls = [a.isometric_image for a in top_assemblies]

    idx, best_img = select(to_build, image_ls, structure_dir=structure_directory)

    return top_assemblies[idx]
