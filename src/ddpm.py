import argparse
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
from PIL import Image
import tqdm


def main(args):
    model = Unet(
        dim=args.model_dim,
        channels=args.channels,
        dim_mults=tuple(args.dim_mults),
        flash_attn=args.flash_attn,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
    )

    trainer = Trainer(
        diffusion,
        args.dataset_path,
        results_folder=args.results_folder,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        amp=args.amp,
        calculate_fid=args.calculate_fid,
        save_and_sample_every=args.save_and_sample_every,
        convert_image_to=args.convert_image_to,
    )

    if not args.is_sample:
        trainer.train()
    else:
        trainer.load(6)  # load model-6.pt # load the checkpoint

        samples_root = args.sample_outdir
        os.makedirs(samples_root, exist_ok=True)
        len_samples = len(os.listdir(samples_root))

        for epoch in tqdm.tqdm(list(range(args.sample_epochs))):
            sampled_images = diffusion.sample(batch_size=args.sample_batchsize)
            for i in range(sampled_images.size(0)):
                current_image_tensor = sampled_images[i]
                current_image = Image.fromarray(
                    (current_image_tensor[0].cpu().numpy() * 255).astype("uint8")
                )
                file_name = f"output__image_{epoch}_{i + len_samples}.png"
                current_image.save(os.path.join(samples_root, file_name))
        print("all samples are save in folder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a diffusion model with specified hyperparameters."
    )

    # Model hyperparameters
    parser.add_argument("--model_dim", type=int, default=64, help="Model dimension.")
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of channels in the input."
    )
    parser.add_argument(
        "--dim_mults",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Dimension multipliers for the U-Net.",
    )
    parser.add_argument(
        "--flash_attn", action="store_true", help="Enable flash attention."
    )

    # Diffusion hyperparameters
    parser.add_argument(
        "--image_size", type=int, default=256, help="Size of the input image."
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of timesteps for diffusion."
    )
    parser.add_argument(
        "--sampling_timesteps",
        type=int,
        default=500,
        help="Number of sampling timesteps.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset."
    )

    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--train_lr", type=float, default=8e-5, help="Learning rate for training."
    )
    parser.add_argument(
        "--train_num_steps", type=int, default=300000, help="Number of training steps."
    )
    parser.add_argument(
        "--gradient_accumulate_every",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.995,
        help="Exponential moving average decay.",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision."
    )
    parser.add_argument(
        "--calculate_fid",
        action="store_true",
        help="Calculate FID during training.",
    )
    parser.add_argument(
        "--save_and_sample_every",
        type=int,
        default=300000,
        help="Frequency for saving and sampling.",
    )
    parser.add_argument(
        "--convert_image_to",
        type=str,
        default="L",
        help="Convert images to a specific format (e.g., 'L' for grayscale).",
    )

    # Sampling hyperparameters
    parser.add_argument(
        "--is_sample",
        action="store_true",
        help="Do sampling or training",
    )
    parser.add_argument(
        "--sample_outdir",
        type=str,
        required=True,
        help="Path to the output directory for generated images.",
    )
    parser.add_argument(
        "--sample_epochs",
        type=int,
        default=1000,
        help="Number for sampling epochs.",
    )
    parser.add_argument(
        "--sample_batchsize",
        type=int,
        default=32,
        help="Batch size for sampling.",
    )

    args = parser.parse_args()
    print(args)
    main(args)
