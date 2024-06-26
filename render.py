#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import ast
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import copy
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def process_viewer_cam(sample_cam, new_pose):
    assert len(new_pose) == 16, "new input camera pose has to be length 16"
    rot_3x4 = np.array(new_pose[:12]).reshape(3, 4)
    rot_3x3 = rot_3x4[:, :-1]
    trans = new_pose[-4:-1]
    new_cam = copy.deepcopy(sample_cam)
    new_cam.R = rot_3x3
    new_cam.T = trans
    return new_cam

def validate_cams(lists_of_numbers, length):
    if not isinstance(lists_of_numbers, list):
        raise ValueError("Input is not a list.")
    for sublist in lists_of_numbers:
        if not isinstance(sublist, list):
            raise ValueError("Sublist is not a list.")
        if len(sublist) != length:
            raise ValueError(f"Sublist {sublist} does not have the required length of {length}.")
        for num in sublist:
            if not isinstance(num, (int, float)):
                raise ValueError(f"Element {num} in sublist {sublist} is not a number.")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")\

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        rendering = rendering[:3, :, :]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_novel(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor, new_cams):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"novel_view_{scale_factor}")

    makedirs(render_path, exist_ok=True)

    for idx, new_cam in enumerate(tqdm(new_cams, desc="Rendering progress")):

        view = process_viewer_cam(views[0], new_cam)
        view.reprocess_cam()

        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        rendering = rendering[:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, new_cams : list):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)
        
        if new_cams is not None:
            render_novel(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor, new_cams=new_cams)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        '--new_cams',
        type=str,
        default=None,
        help="A list of lists of numbers, e.g. '[[1,2,3], [3,4,5]]'"
    )

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    new_cams = ast.literal_eval(args.new_cams)
    try:
        validate_cams(new_cams, 16)
        print("Validated list of lists:", new_cams)
    except ValueError as e:
        print("Validation Error:", e)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, new_cams)