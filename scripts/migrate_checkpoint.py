"""
Migrate checkpoints from old nn.TransformerEncoder format to new TransformerBlock format.

Key mappings:
  OLD (nn.TransformerEncoder)              → NEW (TransformerBlock)
  transformer.layers.{i}.self_attn.in_proj_weight/bias → layers.{i}.wqkv.weight/bias
  transformer.layers.{i}.self_attn.out_proj.weight/bias → layers.{i}.wo.weight/bias
  transformer.layers.{i}.linear1.weight/bias   → layers.{i}.ff.0.weight/bias
  transformer.layers.{i}.linear2.weight/bias   → layers.{i}.ff.2.weight/bias
  transformer.layers.{i}.norm1.weight/bias     → layers.{i}.ln1.weight/bias
  transformer.layers.{i}.norm2.weight/bias     → layers.{i}.ln2.weight/bias
  poke_mask                                    → attn_bias (bool→float)

Usage:
  python scripts/migrate_checkpoint.py <input.pt> <output.pt>
  python scripts/migrate_checkpoint.py checkpoints/bc_best.pt checkpoints/bc_best_migrated.pt
"""

import argparse
import sys
from pathlib import Path

import torch


def migrate_state_dict(old_dict: dict) -> dict:
    """Convert old nn.TransformerEncoder state dict to new TransformerBlock format."""
    new_dict = {}

    for key, value in old_dict.items():
        new_key = key

        # transformer.layers.{i}.self_attn.in_proj_weight → layers.{i}.wqkv.weight
        if "transformer.layers." in key:
            new_key = key.replace("transformer.layers.", "layers.")
            new_key = new_key.replace(".self_attn.in_proj_weight", ".wqkv.weight")
            new_key = new_key.replace(".self_attn.in_proj_bias", ".wqkv.bias")
            new_key = new_key.replace(".self_attn.out_proj.weight", ".wo.weight")
            new_key = new_key.replace(".self_attn.out_proj.bias", ".wo.bias")
            new_key = new_key.replace(".linear1.weight", ".ff.0.weight")
            new_key = new_key.replace(".linear1.bias", ".ff.0.bias")
            new_key = new_key.replace(".linear2.weight", ".ff.2.weight")
            new_key = new_key.replace(".linear2.bias", ".ff.2.bias")
            new_key = new_key.replace(".norm1.weight", ".ln1.weight")
            new_key = new_key.replace(".norm1.bias", ".ln1.bias")
            new_key = new_key.replace(".norm2.weight", ".ln2.weight")
            new_key = new_key.replace(".norm2.bias", ".ln2.bias")

        # poke_mask (bool) → attn_bias (float)
        elif key == "poke_mask":
            # Old: (15, 15) bool, True = blocked
            # New: (1, 1, 15, 15) float, -inf = blocked, 0 = allowed
            n = value.shape[0]
            attn_bias = torch.zeros(1, 1, n, n)
            attn_bias[0, 0, value] = float("-inf")
            new_dict["attn_bias"] = attn_bias
            continue  # don't add old key

        new_dict[new_key] = value

    return new_dict


def migrate_checkpoint(old_path: str, new_path: str):
    """Load checkpoint, migrate state dict, save to new path."""
    print(f"Loading checkpoint: {old_path}")
    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)

    # Handle various checkpoint formats
    sd_key = None
    for k in ("model_state_dict", "model_state", "state_dict"):
        if k in checkpoint:
            sd_key = k
            break

    if sd_key:
        print(f"Found '{sd_key}' in checkpoint")
        checkpoint[sd_key] = migrate_state_dict(checkpoint[sd_key])
    else:
        # Assume the whole thing is a state dict
        print("Treating entire file as state dict")
        checkpoint = migrate_state_dict(checkpoint)

    print(f"Saving migrated checkpoint: {new_path}")
    torch.save(checkpoint, new_path)
    print("Done!")


def verify_migration(old_path: str, new_path: str):
    """Verify that migrated checkpoint can be loaded by new model."""
    from pokebot.model.poke_transformer import PokeTransformer

    checkpoint = torch.load(new_path, map_location="cpu", weights_only=False)

    sd = checkpoint
    for k in ("model_state_dict", "model_state", "state_dict"):
        if isinstance(checkpoint, dict) and k in checkpoint:
            sd = checkpoint[k]
            break

    model = PokeTransformer()
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")
    if not missing and not unexpected:
        print("Verification PASSED: all keys match perfectly")

    # Quick forward pass test
    B = 2
    int_ids = torch.zeros(B, 15, 8, dtype=torch.long)
    float_feats = torch.randn(B, 15, 394)
    legal_mask = torch.ones(B, 10)

    with torch.no_grad():
        log_probs, value_probs, value = model(int_ids, float_feats, legal_mask)

    assert not torch.isnan(log_probs).any(), "NaN in log_probs!"
    assert not torch.isnan(value).any(), "NaN in value!"
    print(f"Forward pass OK: log_probs={log_probs.shape}, value={value.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate checkpoint to new format")
    parser.add_argument("input", help="Input checkpoint path")
    parser.add_argument("output", help="Output checkpoint path")
    parser.add_argument("--verify", action="store_true", help="Verify after migration")
    args = parser.parse_args()

    migrate_checkpoint(args.input, args.output)

    if args.verify:
        print("\n--- Verification ---")
        verify_migration(args.input, args.output)
