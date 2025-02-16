from bloxnet.pipelines.whole_structure import (
    get_structure_info,
    get_structure_rating,
    SAVE_DIR,
)
import os

from bloxnet.utils.utils import get_last_json_as_dict, save_to_json, slugify
from bloxnet.prompts.prompt import prompt_with_caching
from bloxnet.structure import Assembly


def _get_ratings(to_build, isometric_img, iter=0):
    to_build_slug = slugify(to_build)
    structure_dir = os.path.join(SAVE_DIR, to_build_slug)

    # eval
    prompt = get_structure_info(isometric_img)
    response, eval_context = prompt_with_caching(
        prompt, [], structure_dir, "info_eval", cache=True, i=iter
    )

    json_guesses = get_last_json_as_dict(response)
    save_to_json(json_guesses, os.path.join(SAVE_DIR, to_build_slug, "guesses.json"))

    # get rating
    prompt = get_structure_rating(to_build)
    response, eval_context = prompt_with_caching(
        prompt, eval_context, structure_dir, "rating_eval", cache=True, i=iter
    )

    json_rating = get_last_json_as_dict(response)
    save_to_json(json_rating, os.path.join(SAVE_DIR, to_build_slug, "rating.json"))

    return json_guesses["guesses"], json_rating["rating"]


def get_ratings(assembly: Assembly):
    """
    Populates rating and guesses attribute of assembly input
    """
    guesses, rating = _get_ratings(
        assembly.to_build, assembly.isometric_image, iter=assembly.assembly_num
    )

    assembly.eval_guesses = guesses
    assembly.eval_rating = rating
