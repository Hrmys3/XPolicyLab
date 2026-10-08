import os

# Set rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"

import torch
import numpy as np
import pickle
import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from utils import load_data  # data functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from detr.act_policy import ACTPolicy, CNNMLPPolicy

import IPython

e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    ckpt_setting = args["ckpt_setting"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]

    # get task parameters
    from constants import TASK_CONFIGS

    task_config = TASK_CONFIGS[ckpt_setting]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = int(os.environ.get("ACT_ACTION_DIM"))
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "ckpt_setting": ckpt_setting,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "save_freq": args['save_freq'],
        "init_ckpt": args.get("init_ckpt"),
    }

    train_dataloader, val_dataloader, stats, _, train_indices, val_indices = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, "policy_config.json"), "w", encoding="utf-8") as f:
        json.dump({"chunk_size": args["chunk_size"], "hidden_dim": args["hidden_dim"],
                   "dim_feedforward": args["dim_feedforward"], "kl_weight": args["kl_weight"],
                   "camera_names": camera_names}, f, indent=2)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    with open(os.path.join(ckpt_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_seed": 1,
                "train_episode_ids": train_indices.tolist(),
                "val_episode_ids": val_indices.tolist(),
                "init_ckpt": args.get("init_ckpt"),
            },
            f,
            indent=2,
        )
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    # ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    # torch.save(best_state_dict, ckpt_path)
    # print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config["init_ckpt"]:
        policy.load_state_dict(torch.load(config["init_ckpt"], map_location="cpu", weights_only=True))
        print(f"Initialized policy weights from {config['init_ckpt']} (optimizer starts fresh)")
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None
    metrics_path = os.path.join(ckpt_dir, "metrics.jsonl")

    progress = tqdm(range(num_epochs))
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for epoch in progress:
            policy.train()
            train_batches = []
            for data in train_dataloader:
                optimizer.zero_grad()
                forward_dict = forward_pass(data, policy)
                loss = forward_dict["loss"]
                loss.backward()
                optimizer.step()
                train_batches.append(detach_dict(forward_dict))
            train_summary = compute_dict_mean(train_batches)

            with torch.inference_mode():
                policy.eval()
                val_batches = []
                for data in val_dataloader:
                    forward_dict = forward_pass(data, policy)
                    val_batches.append(forward_dict)
                val_summary = compute_dict_mean(val_batches)

            epoch_val_loss = float(val_summary["loss"].item())
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch + 1, min_val_loss, deepcopy(policy.state_dict()))

            metrics = {"epoch": epoch + 1}
            metrics.update({f"train_{key}": float(value.item()) for key, value in train_summary.items()})
            metrics.update({f"val_{key}": float(value.item()) for key, value in val_summary.items()})
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            progress.set_postfix(train=f"{metrics['train_loss']:.3f}", val=f"{epoch_val_loss:.3f}")

            if (epoch + 1) % config['save_freq'] == 0:
                ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
                torch.save(policy.state_dict(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)
    best_epoch, _, best_state_dict = best_ckpt_info
    torch.save(best_state_dict, os.path.join(ckpt_dir, "policy_best.ckpt"))
    print(f"Best validation loss: {min_val_loss:.6f} at epoch {best_epoch}")

    return best_ckpt_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--ckpt_setting", action="store", type=str, help="ckpt_setting", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--save_freq", action="store", type=int, help="save ckpt frequency", required=False, default=6000)
    parser.add_argument("--init_ckpt", type=str, default=None, help="Initialize model weights from a checkpoint; optimizer starts fresh")
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")

    main(vars(parser.parse_args()))
