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
from taylor_utils import interpolate_features



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



    coarse_steps = args.num_sampling_steps // args.interval + 1
    diffusion_fine = create_diffusion(str(args.num_sampling_steps+1))
    diffusion_coarse = create_diffusion(str(coarse_steps))

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


    z_coarse = z.clone()
    z_fine = z.clone()
    prev_z = None
    prev_features = None
    prevprev_features = None 

    num_segments = coarse_steps - 1
    interval = args.interval


    interp_time_total = 0.0
    fine_time_total = 0.0
    coarse_time_total = 0.0

    ## 저장용
    attn_interp_all = []
    mlp_interp_all = []
    attn_features_all = []
    mlp_features_all = []


    ##################

    if args.ddim_sample:
        if args.use_interp:
            for seg in range(coarse_steps):

                torch.cuda.synchronize()
                coarse_start = torch.cuda.Event(enable_timing=True)
                coarse_end = torch.cuda.Event(enable_timing=True)
                coarse_start.record()
                if seg == 0:
                    pass
                else:
                    print(f"\n===== [Segment {seg}] Coarse {seg}->{seg+1}, Fine {(seg-1)*interval}->{seg*interval} =====")

                # coarse anchor forward ===============
                model.feature_collection_mode = True
                model.interpolation_mode = False
                model.interpolation_mode_coarse = False
                
                model.collected_attn_features = []
                model.collected_mlp_features = []

                for block in model.blocks:
                    block.collected_attn_features = model.collected_attn_features
                    block.collected_mlp_features = model.collected_mlp_features
                
                z_coarse = diffusion_coarse.ddim_sample_loop(
                    model.forward_with_cfg,
                    z_coarse.shape,
                    noise=z_coarse,  # coarse chain 유지
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                    start_step=seg,
                    end_step=seg + 1
                )

                coarse_end.record()
                torch.cuda.synchronize()
                coarse_elapsed = coarse_start.elapsed_time(coarse_end) * 0.001
                coarse_time_total += coarse_elapsed

                curr_features = {
                    "attn": torch.stack(model.collected_attn_features, dim=0),
                    "mlp":  torch.stack(model.collected_mlp_features, dim=0),
                }


                ####저장용
                # attn_features_all.append(curr_features["attn"].detach().cpu())
                # mlp_features_all.append(curr_features["mlp"].detach().cpu())

                curr_z = z_coarse  # 현재 coarse latent

                if prev_features is not None:

                    total_steps = (coarse_steps - 1) * interval
                    stage_ratio = (seg * interval) / total_steps
                    stage_ratio = min(max(stage_ratio, 0.0), 1.0)
                    
                    # ① Interpolation
                    attn_interp_ = interpolate_features([prev_features["attn"], curr_features["attn"]],
                                                    target_T=interval+1,
                                                    prevprev_tensor=None if prevprev_features is None else prevprev_features["attn"],
                                                    stage_ratio=stage_ratio)
                    mlp_interp_ = interpolate_features([prev_features["mlp"], curr_features["mlp"]],
                                                    target_T=interval+1,
                                                    prevprev_tensor=None if prevprev_features is None else prevprev_features["mlp"],
                                                    stage_ratio=stage_ratio)
                    
                    attn_interp = attn_interp_[:-1]
                    mlp_interp = mlp_interp_[:-1]

                    attn_interp_coarse = attn_interp_[-1].unsqueeze(0)
                    mlp_interp_coarse = mlp_interp_[-1].unsqueeze(0)

                    # ② Fine inference (interpolation 기반)
                    model.feature_collection_mode = False
                    model.interpolation_mode = True
                    model.interpolation_mode_coarse = False
                    model.interpolated_attn_features = attn_interp
                    model.interpolated_mlp_features = mlp_interp

                    z_fine = diffusion_fine.ddim_sample_loop(
                        model.forward_with_cfg,
                        z_fine.shape,
                        noise=z_fine,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False,
                        device=device,
                        start_step=(seg-1)*interval,
                        end_step=seg*interval
                    )

                    # ==========================
                    # ③ Corrector: fine latent를 coarse chain에 반영 / last step anchor computation
                    # ==========================
                    model.feature_collection_mode = False
                    model.interpolation_mode = False

                    model.collected_attn_features = []
                    model.collected_mlp_features = []
                    for block in model.blocks:
                        block.collected_attn_features = model.collected_attn_features
                        block.collected_mlp_features = model.collected_mlp_features

                    # fine output을 coarse segment의 start로 설정
                    z_coarse = z_fine.clone()

                    model.interpolation_mode_coarse = True
                    model.coarse_feature_attn = attn_interp_coarse
                    model.coarse_feature_mlp = mlp_interp_coarse

                    if seg == (coarse_steps - 1):
                        print("Final computation")
                        z_fine = diffusion_fine.ddim_sample_loop(
                            model.forward_with_cfg,
                            z_fine.shape,
                            noise=z_fine,
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            progress=False,
                            device=device,
                            start_step=seg*interval,
                            end_step=seg*interval+1
                        )         
                    else:               
                        # coarse corrector forward (1->2를 다시 수행)
                        z_coarse = diffusion_coarse.ddim_sample_loop(
                            model.forward_with_cfg,
                            z_coarse.shape,
                            noise=z_coarse,
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            progress=False,
                            device=device,
                            start_step=seg,
                            end_step=seg + 1
                        )

                    # # Corrected coarse features 저장
                    # corrected_features = {
                    #     "attn": torch.stack(model.collected_attn_features, dim=0),
                    #     "mlp": torch.stack(model.collected_mlp_features, dim=0),
                    # }

                    #prevprev_features = prev_features
                    #prev_features = corrected_features
                    prev_features = curr_features  

                else:
                    prev_features = curr_features

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
        fine_steps_run = len(diffusion_fine.x0_predictions)
        print(f"[DEBUG] fine steps actually run = {fine_steps_run} (interval={interval})")

    print(f"\n[STATS] Total coarse forward time = {coarse_time_total:.4f} sec")
    print(f"[STATS] Total fine forward time   = {fine_time_total:.4f} sec")
    print(f"[STATS] Total interpolation time  = {interp_time_total:.4f} sec")
    print(f"[STATS] Total sampling time (coarse+fine+interp) = {coarse_time_total + fine_time_total + interp_time_total:.4f} sec")
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
        # if len(attn_interp_all) > 0:
        #     torch.save(attn_interp_all, os.path.join(args.sample_dir, "attn_interpolated_features.pt"))
        # if len(mlp_interp_all) > 0:
        #     torch.save(mlp_interp_all, os.path.join(args.sample_dir, "mlp_interpolated_features.pt"))


        # ### COARSE FEATURES
        # if attn_features_all:
        #     torch.save(attn_features_all, os.path.join(args.sample_dir, "attn_features_coarse.pt"))
        # if mlp_features_all:
        #     torch.save(mlp_features_all, os.path.join(args.sample_dir, "mlp_features_coarse.pt"))
            

        # ### FULL INFERENCE features
        # for block in model.blocks:
        #     if hasattr(block, 'attn_features') and len(block.attn_features) > 0:
        #         attn_features.append(torch.stack(block.attn_features))
        #     if hasattr(block, 'mlp_features') and len(block.mlp_features) > 0:
        #         mlp_features.append(torch.stack(block.mlp_features))

        # if attn_features:
        #     torch.save(attn_features, os.path.join(args.sample_dir, "attn_features_full.pt"))
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
            torch.save(pred_noises, os.path.join(args.sample_dir, "interpolation_noise2.pt"))

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

