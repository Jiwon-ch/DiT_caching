# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os
import numpy as np
from taylor_utils import interpolate_features, get_interval
import math


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed) #일시적으로 다른 image 생성
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu" 
    #print("device = ", device, flush=True)
    #print(torch.cuda.device_count(), flush=True)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    model.use_interp = args.use_interp
    #model.interpolation_mode = args.use_interp 

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    #ckpt_path = args.ckpt or f"/root/autodl-tmp/pretrained_models/DiT/DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    ckpt_path = args.ckpt or f"pretrained_models/DiT-XL-2-{args.image_size}x{args.image_size}.pt"

    
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    #vae = AutoencoderKL.from_pretrained(f"/root/autodl-tmp/pretrained_models/stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    #class_labels = [207, 360, 387, 974, 88, 979, 417, 279,]
    #class_labels = [985, 130, 987, 130, 292, 289, 339, 385, 293, 397, 974, 814]
    # change ID number 15 to any other ImageNet category ID
    #class_labels = [30]
    #class_labels = [30, 309, 311, 974, 497]
    class_labels = args.class_labels

    # Create sampling noise:
    n = len(class_labels)
    # Sample 4 images for category label
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    #print("cfg scale = ", args.cfg_scale, flush=True)
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    model_kwargs['interval']        = args.interval
    model_kwargs['max_order']       = args.max_order
    model_kwargs['test_FLOPs']      = args.test_FLOPs

    model_kwargs['use_interp'] = args.use_interp

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()


    coarse_steps = math.ceil(args.num_sampling_steps / args.interval) 
    total_steps = args.num_sampling_steps
    z_coarse = z.clone()
    z_fine = z.clone()
    prev_z = None
    prev_features = None
    prevprev_features = None 


    interval = args.interval
    interval_prev = None
    cut_t = 0

    correction_time_total = 0.0
    fine_time_total = 0.0
    coarse_time_total = 0.0
    interp_time_total = 0.0

    ## 저장용
    attn_interp_all = []
    mlp_interp_all = []
    attn_features_all = []
    mlp_features_all = []
    

    
    ##################

    if args.ddim_sample:
        if args.use_interp:
            while cut_t < total_steps:
                start_t = cut_t
                end_t = min(cut_t + interval , total_steps - 1)

                print(f"\n[COARSE {start_t}->{end_t}] (interval={interval})")            
                model.feature_collection_mode = True
                model.interpolation_mode = False
                model.interpolation_mode_coarse = False

                model.collected_attn_features, model.collected_mlp_features = [], []
                for block in model.blocks:
                    block.collected_attn_features = model.collected_attn_features
                    block.collected_mlp_features = model.collected_mlp_features

                skip = interval if (end_t - start_t) >= interval else 1

                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                z_coarse = diffusion.ddim_sample_loop(
                    model.forward_with_cfg,
                    z_coarse.shape,
                    noise=z_coarse,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                    start_step=start_t,
                    end_step=end_t, 
                    skip_step=skip 
                )

                end_event.record()
                torch.cuda.synchronize()
                coarse_time = start_event.elapsed_time(end_event) * 0.001
                coarse_time_total += coarse_time

                curr_features = {
                    "attn": torch.stack(model.collected_attn_features, dim=0),
                    "mlp":  torch.stack(model.collected_mlp_features, dim=0),
                }

                if end_t >= total_steps - 1:
                    prev_interval = interval_prev if interval_prev is not None else interval
                    print(f"\n[Last FINE {start_t-interval}->{total_steps}")

                    torch.cuda.synchronize()
                    start_event.record()

                    attn_interp_ = interpolate_features(
                        [prev_features["attn"], curr_features["attn"]],
                        target_T=  prev_interval + 1,
                        prevprev_tensor=None if prevprev_features is None else prevprev_features["attn"],
                        stage_ratio=stage_ratio
                    )
                    mlp_interp_ = interpolate_features(
                        [prev_features["mlp"], curr_features["mlp"]],
                        target_T= prev_interval + 1,
                        prevprev_tensor=None if prevprev_features is None else prevprev_features["mlp"],
                        stage_ratio=stage_ratio
                    )

                    attn_interp, mlp_interp = attn_interp_[:-1], mlp_interp_[:-1]

                    #attn_interp_all.append(attn_interp_.detach().cpu())

                    model.interpolated_attn_features = attn_interp
                    model.interpolated_mlp_features = mlp_interp


                    model.feature_collection_mode = False
                    model.interpolation_mode = True
                    model.interpolation_mode_coarse = False

                    z_fine = diffusion.ddim_sample_loop(
                        model.forward_with_cfg,
                        z_fine.shape,
                        noise=z_fine,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False,
                        device=device,
                        start_step=start_t-prev_interval,
                        end_step=start_t
                    )

                    model.feature_collection_mode = False
                    model.interpolation_mode = False
                    model.interpolation_mode_coarse = False

                    z_fine = diffusion.ddim_sample_loop(
                        model.forward_with_cfg,
                        z_fine.shape,
                        noise=z_fine,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False,
                        device=device,
                        start_step=start_t,
                        end_step=total_steps
                    )

                    end_event.record()
                    torch.cuda.synchronize()
                    fine_time = start_event.elapsed_time(end_event) * 0.001
                    fine_time_total += fine_time

                    break

                # === (2) If this is not the last segment, interpolate + fine ===
                if end_t < total_steps - 1 and prev_features is not None:
                    prev_interval = interval_prev if interval_prev is not None else interval
                    print(f"\n[Segment FINE {start_t-interval}->{start_t-1}")

                    stage_ratio = start_t / (total_steps - interval)
                    stage_ratio = min(max(stage_ratio, 0.0), 1.0)

                    attn_interp_ = interpolate_features(
                        [prev_features["attn"], curr_features["attn"]],
                        target_T=prev_interval + 1,
                        prevprev_tensor=None if prevprev_features is None else prevprev_features["attn"],
                        stage_ratio=stage_ratio
                    )
                    mlp_interp_ = interpolate_features(
                        [prev_features["mlp"], curr_features["mlp"]],
                        target_T=prev_interval + 1,
                        prevprev_tensor=None if prevprev_features is None else prevprev_features["mlp"],
                        stage_ratio=stage_ratio
                    )

                    attn_interp, mlp_interp = attn_interp_[:-1], mlp_interp_[:-1]


                    #attn_interp_all.append(attn_interp.detach().cpu())


                    attn_interp_coarse = attn_interp_[-1].unsqueeze(0)
                    mlp_interp_coarse = mlp_interp_[-1].unsqueeze(0)

                    # === (3) Fine inference for this segment ===
                    model.feature_collection_mode = False
                    model.interpolation_mode = True
                    model.interpolation_mode_coarse = False

                    model.interpolated_attn_features = attn_interp
                    model.interpolated_mlp_features = mlp_interp

                    torch.cuda.synchronize()
                    start_event.record()

                    z_fine = diffusion.ddim_sample_loop(
                        model.forward_with_cfg,
                        z_fine.shape,
                        noise=z_fine,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False,
                        device=device,
                        start_step=start_t-prev_interval,
                        end_step=start_t
                    )

                    end_event.record()
                    torch.cuda.synchronize()
                    fine_time = start_event.elapsed_time(end_event) * 0.001
                    fine_time_total += fine_time

                    ###############################
                    ###### Correction Step ########
                    ###############################

                    model.feature_collection_mode = False
                    model.interpolation_mode = False
                    model.interpolation_mode_coarse = True
                    
                    model.collected_attn_features = []
                    model.collected_mlp_features = []

                    for block in model.blocks:
                        block.collected_attn_features = model.collected_attn_features
                        block.collected_mlp_features = model.collected_mlp_features
                    
                    z_coarse = z_fine.clone()


                    model.coarse_feature_attn = attn_interp_coarse
                    model.coarse_feature_mlp = mlp_interp_coarse


                    torch.cuda.synchronize()
                    start_event.record()
                    
                    print(f"[CORRECTION {start_t}->{end_t}] (new interval={interval})")

                    z_coarse = diffusion.ddim_sample_loop(
                        model.forward_with_cfg,
                        z_coarse.shape,
                        noise=z_coarse,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False,
                        device=device,
                        start_step=start_t,
                        end_step=end_t,
                        skip_step=interval
                    )
                    
                    end_event.record()
                    torch.cuda.synchronize()
                    correction_time = start_event.elapsed_time(end_event) * 0.001
                    correction_time_total += correction_time


                    new_interval = get_interval(prev_features, curr_features, interval)      

                    interval_prev = interval
                    interval = new_interval  
                    prev_features = curr_features 
                else:
                    prev_features = curr_features
                
                cut_t = end_t

        else: 
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, 
                model_kwargs=model_kwargs, progress=True, device=device
            )
    else:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

    if args.use_interp:
        samples = z_fine
        print(f"Total coarse forward time = {coarse_time_total:.3f} sec")
        print(f"Total fine forward time   = {fine_time_total:.3f} sec")
        print(f"Total correction time     = {correction_time_total:.3f} sec")
        print(f"Total sampling time (all) = {coarse_time_total + fine_time_total + correction_time_total:.3f} sec")
        
    end.record()
    torch.cuda.synchronize()
    print(f"Total Sampling took {start.elapsed_time(end)*0.001} seconds")

    ########@@@@@@@@@@@@@@@ features
    if args.save:
        attn_features = []
        mlp_features = []


        x0_predictions = []
        pred_noises = []
        cond_noises = []
        uncond_noises = []

        # ## INTERPOLATION FEATURES
        if len(attn_interp_all) > 0:
            torch.save(attn_interp_all, os.path.join(args.sample_dir, "attn_interpolated_features.pt"))
        # if len(mlp_interp_all) > 0:
        #     torch.save(mlp_interp_all, os.path.join(args.sample_dir, "mlp_interpolated_features.pt"))


        # ### COARSE FEATURES
        # if attn_features_all:
        #     torch.save(attn_features_all, os.path.join(args.sample_dir, "attn_features_coarse.pt"))
        # if mlp_features_all:
        #     torch.save(mlp_features_all, os.path.join(args.sample_dir, "mlp_features_coarse.pt"))
            

        # ### FULL INFERENCE features
        for block in model.blocks:
            if hasattr(block, 'attn_features') and len(block.attn_features) > 0:
                attn_features.append(torch.stack(block.attn_features))
        #     if hasattr(block, 'mlp_features') and len(block.mlp_features) > 0:
        #         mlp_features.append(torch.stack(block.mlp_features))

        if attn_features:
            torch.save(attn_features, os.path.join(args.sample_dir, "attn_features_full1.pt"))
        # if mlp_features:
        #     torch.save(mlp_features, os.path.join(args.sample_dir, "mlp_features_full.pt"))
        ##########
            

        # if hasattr(diffusion, 'x0_predictions'):
        #     x0_predictions = diffusion.x0_predictions       
        # if hasattr(model, 'cond_noises'):
        #     cond_noises = model.cond_noises
        # if hasattr(model, 'uncond_noises'):
        #     uncond_noises = model.uncond_noises
        if hasattr(model, 'pred_noises'):
            pred_noises = model.pred_noises
            
        # if x0_predictions: 
        #     torch.save(x0_predictions, os.path.join(args.sample_dir, "x0_predictions.pt"))        
        # if cond_noises:
        #     torch.save(cond_noises, os.path.join(args.sample_dir, "cond_noises.pt"))
        # if uncond_noises:
        #     torch.save(uncond_noises, os.path.join(args.sample_dir, "uncond_noises.pt"))
        if pred_noises:
            torch.save(pred_noises, os.path.join(args.sample_dir, "noise1.pt"))

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample_1.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--ddim-sample", action="store_true", default=False)
    parser.add_argument("--interval", type=int, default=4) 
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--test-FLOPs", action="store_true", default=False)
    parser.add_argument("--sample_dir", type=str, default="noises")
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--use_interp", action="store_true", help="Enable interpolation mode")
    parser.add_argument("--class-labels", type=int, nargs="+", default=[30])
    # parser.add_argument("--interp-method", type=str, 
    #                 choices=['taylor', 'cubic_spline', 'pchip', 'polynomial', 'bezier'], 
    #                 default='linear')
    #parser.add_argument("--merge-weight", type=float, default=0.0) # never used in toca, just for exploration

    args = parser.parse_args()
    main(args)

