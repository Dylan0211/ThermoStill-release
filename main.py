"""Command-line entrypoint for ThermoStill."""

import argparse
import importlib.util
import logging
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from exp.exp_thermostill import Exp_ThermoStill

warnings.filterwarnings("ignore")


class TeeStream:
    """Write stdout/stderr to both console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_parser() -> argparse.ArgumentParser:
    """Define the public CLI for the ThermoStill training entrypoint."""
    parser = argparse.ArgumentParser(description="ThermoStill")

    parser.add_argument("--file_name", type=str, default="house_id_da09897f6b67c4511ee33c658ddbdfe3afd082e3.csv")
    parser.add_argument("--state_dataset", type=str, default="TX")
    parser.add_argument(
        "--rc_model",
        type=str,
        default="R1C1",
        choices=["R1C1", "R2C1", "R2C2"],
    )
    parser.add_argument(
        "--tsfm_name_list",
        nargs="+",
        type=str,
        default=["chronos", "timesfm", "timemoe"],
        choices=["chronos", "timesfm", "timemoe"],
    )
    parser.add_argument("--train_days", type=int, default=24)
    parser.add_argument("--val_days", type=int, default=4)
    parser.add_argument("--test_days", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=168)
    parser.add_argument("--prediction_length", type=int, default=1)
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--delta_t", type=float, default=3600.0)
    parser.add_argument("--rc_lr", type=float, default=1e-3)
    parser.add_argument("--policy_lr", type=float, default=5e-6)
    parser.add_argument("--value_lr", type=float, default=5e-6)
    parser.add_argument("--max_ppo_epochs", type=int, default=3)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--input_enc_dim", type=int, default=64)
    parser.add_argument("--trend_kernel_size", type=int, default=25)

    parser.add_argument("--alpha_m", type=float, default=0.5, help="KD weight lambda in the paper")
    parser.add_argument("--reward_eta", type=float, default=0.8)
    parser.add_argument("--phy_reg_weight", type=float, default=1.0)

    parser.add_argument(
        "--device",
        type=str,
        default=str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    )
    parser.add_argument("--scale", type=bool, default=True)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--lradj", type=str, default="type1")

    parser.add_argument("--project_root", default=Path(__file__).parent)
    parser.add_argument("--graph_dir", type=str, default="graphs")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dataset_raw_dir", type=str, default="data/raw/ecobee/house_data_by_state")
    parser.add_argument("--tmp_data_dir", type=str, default="tmp_data")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    return parser


def set_seed(seed: int = 2021) -> None:
    """Set the default seed used by all training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_runtime_dependencies(args) -> None:
    module_by_teacher = {
        "chronos": "chronos",
        "timesfm": "timesfm",
        "timemoe": "transformers",
    }
    missing = []
    for teacher in args.tsfm_name_list:
        module_name = module_by_teacher.get(teacher, teacher)
        if importlib.util.find_spec(module_name) is None:
            missing.append((teacher, module_name))
    if missing:
        detail = ", ".join([f"{teacher} (import '{module}')" for teacher, module in missing])
        raise ModuleNotFoundError(
            "Missing teacher dependencies: "
            f"{detail}. Install required packages first (e.g. `pip install -r requirements.txt`)."
        )


def _run_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem if stem.startswith("house_id_") else f"house_id_{stem}"


def configure_logging(args) -> Path:
    log_root = Path(args.project_root) / args.log_dir / args.exp_name
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{args.rc_model}_{_run_stem(args.file_name)}.log"

    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(sys.__stdout__),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def print_configuration(args, log_path: Path) -> None:
    print()
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    config = vars(args).copy()
    config["device"] = str(config["device"])
    config["project_root"] = str(Path(config["project_root"]).resolve())
    config["log_path"] = str(log_path)
    for key in sorted(config):
        print(f"{key:<18}: {config[key]}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    set_seed(2021)
    args = build_parser().parse_args()
    args.exp_name = "thermostill"
    args.device = torch.device(args.device)
    log_path = configure_logging(args)
    validate_runtime_dependencies(args)
    print_configuration(args, log_path)
    Exp_ThermoStill(args).train()




