# Blox-Net design generation and perturbation repo
This repository contains code for generating Blox-Net designs and performing perturbation analaysis

## High-Level Organization
**scripts**: Files which should be run directly \
**bloxnet**: Core code for design generation. Queries ChatGPT and simulates block placements. \
**perturbation_analysis**: Code for introducing tolerances and improving design feasibility.

## Setup
For CahtGPT queries, add your API key `OPENAI_API_KEY` to a `.env` file in the root folder of the project or export it as an enviorment variable.

## Scripts
**run_pipeline.py** runs a single generation on on object type
**run_pipeline_single_obj_parallel.py** runs multiple generations in parallel on one object type
**full_pipeline.py** runs multiple generations in parallel of multiple object types in parallel (ex: generates 10 trees, 10 houses, 10 cars...)