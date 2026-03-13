import argparse
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


class DeterministicPolicyExporter(th.nn.Module):
    """Expose deterministic actions for ONNX export."""

    def __init__(self, policy: ActorCriticPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return self.policy.get_distribution(observation).distribution.mean


def parse_hidden_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def infer_policy_layout(state_dict: dict, fallback_obs_dim: int, fallback_action_dim: int, fallback_hidden_sizes: list[int]):
    policy_layer_shapes = []
    value_layer_shapes = []

    index = 0
    while True:
        policy_key = f"mlp_extractor.policy_net.{index}.weight"
        value_key = f"mlp_extractor.value_net.{index}.weight"
        if policy_key not in state_dict or value_key not in state_dict:
            break
        policy_layer_shapes.append(tuple(state_dict[policy_key].shape))
        value_layer_shapes.append(tuple(state_dict[value_key].shape))
        index += 2

    if policy_layer_shapes:
        obs_dim = policy_layer_shapes[0][1]
        pi_hidden_sizes = [shape[0] for shape in policy_layer_shapes]
        vf_hidden_sizes = [shape[0] for shape in value_layer_shapes]
    else:
        obs_dim = fallback_obs_dim
        pi_hidden_sizes = fallback_hidden_sizes
        vf_hidden_sizes = fallback_hidden_sizes

    action_dim = fallback_action_dim
    if "action_net.weight" in state_dict:
        action_dim = int(state_dict["action_net.weight"].shape[0])
    elif "log_std" in state_dict:
        action_dim = int(state_dict["log_std"].shape[0])

    return obs_dim, action_dim, {"pi": pi_hidden_sizes, "vf": vf_hidden_sizes}


def load_policy(policy_path: Path, obs_dim: int, action_dim: int, hidden_sizes: list[int]) -> ActorCriticPolicy:
    checkpoint = th.load(policy_path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    inferred_obs_dim, inferred_action_dim, inferred_net_arch = infer_policy_layout(
        state_dict,
        fallback_obs_dim=obs_dim,
        fallback_action_dim=action_dim,
        fallback_hidden_sizes=hidden_sizes,
    )

    policy_kwargs = {
        "observation_space": spaces.Box(low=-np.inf, high=np.inf, shape=(inferred_obs_dim,), dtype=np.float32),
        "action_space": spaces.Box(low=-1.0, high=1.0, shape=(inferred_action_dim,), dtype=np.float32),
        "lr_schedule": lambda _: 0.0,
        "net_arch": inferred_net_arch,
        "activation_fn": th.nn.ReLU,
    }

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        saved_data = checkpoint.get("data", {})
        policy_kwargs.update(
            {
                key: saved_data[key]
                for key in (
                    "observation_space",
                    "action_space",
                    "net_arch",
                    "activation_fn",
                    "ortho_init",
                    "use_sde",
                    "log_std_init",
                    "full_std",
                    "use_expln",
                    "squash_output",
                    "features_extractor_class",
                    "features_extractor_kwargs",
                    "normalize_images",
                    "optimizer_class",
                    "optimizer_kwargs",
                )
                if key in saved_data
            }
        )

    policy = ActorCriticPolicy(**policy_kwargs)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Stable-Baselines3 policy .pth file to ONNX.")
    parser.add_argument("input", type=Path, help="Path to the input .pth file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ONNX path. Defaults to the input path with a .onnx suffix.",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=10,
        help="Observation vector dimension used when the checkpoint does not include constructor metadata.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=14,
        help="Action vector dimension used when the checkpoint does not include constructor metadata.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=parse_hidden_sizes,
        default=[64],
        help="Comma-separated hidden layer sizes fallback when the checkpoint does not expose enough shape information.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".onnx")

    policy = load_policy(input_path, args.obs_dim, args.action_dim, args.hidden_sizes)
    exporter = DeterministicPolicyExporter(policy)
    dummy_obs = th.randn(1, policy.observation_space.shape[0], dtype=th.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    th.onnx.export(
        exporter,
        dummy_obs,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=args.opset,
    )

    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
