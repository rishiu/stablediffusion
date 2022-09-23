import torch
import argparse

# orig_path = "logs/2022-09-02T06-46-25_pokemon_pokemon/checkpoints/epoch=000142.ckpt"
# out_name = "pokemon-ema-only.ckpt"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_ckpt", help="full size checkpoint file")
    parser.add_argument("--output_path", help="filename for ema only checkpoint")
    args = parser.parse_args()

    print(f"loading from {args.original_ckpt}")
    d = torch.load(args.original_ckpt, map_location="cpu")

    new_d = {"state_dict": {}}
    ema_state = {k: v for k, v in d["state_dict"].items() if not k.startswith("model.diffusion_model")}
    new_d["state_dict"] = ema_state

    print(f"saving to {args.output_path}")
    torch.save(new_d, args.output_path)