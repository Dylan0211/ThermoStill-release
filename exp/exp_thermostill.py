"""End-to-end ThermoStill training loop.

This module implements the main training path used by the public project:
- RC student pretraining
- RL-based teacher weighting
- PPO policy updates
- validation-aware reward
- physical-consistency regularization
"""

import os
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

from data_provider.data_factory import data_provider
from models.rl.thermostill_rl import ThermoStillActor, ThermoStillCritic
from utils.tools import adjust_learning_rate, cv_rmse, mae, mse, rmse, select_rc_model


@dataclass
class EpisodeBuffer:
    """Per-episode rollout storage used for PPO updates."""

    batch_x: list
    teacher_gt_error: list
    teacher_student_gap: list
    actions: list
    log_probs: list
    values: list
    rewards: list
    masks: list


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor, gamma: float, lam: float):
    """Compute generalized advantage estimates for the PPO update."""
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for idx in reversed(range(rewards.shape[0])):
        delta = rewards[idx] + gamma * values[idx + 1] * masks[idx] - values[idx]
        gae = delta + gamma * lam * masks[idx] * gae
        advantages[idx] = gae
    returns = advantages + values[:-1]
    return advantages, returns


class Exp_ThermoStill:
    """Experiment runner for the public ThermoStill implementation."""

    def __init__(self, args):
        self.args = args
        self.criterion = nn.MSELoss()

    def _log_header(self, title: str) -> None:
        print()
        print("=" * 60)
        print(title)
        print("=" * 60)

    def _format_metrics_line(self, metrics) -> str:
        return (
            f"one-step mae={metrics[0]:.4f}, mse={metrics[1]:.4f}, rmse={metrics[2]:.4f}, cv_rmse={metrics[3]:.4f}% | "
            f"multi-step mae={metrics[4]:.4f}, mse={metrics[5]:.4f}, rmse={metrics[6]:.4f}, cv_rmse={metrics[7]:.4f}%"
        )

    def _action_summary(self, actions: torch.Tensor) -> str:
        action_mean = actions.mean(dim=0).detach().cpu().tolist()
        pairs = [f"{teacher}={weight:.4f}" for teacher, weight in zip(self.args.tsfm_name_list, action_mean)]
        return ", ".join(pairs)

    def _maybe_save_checkpoint(self, rc_model) -> None:
        output_dir = os.path.join(self.args.project_root, self.args.ckpt_dir, self.args.exp_name)
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"{self.args.rc_model}_{self._run_name()}.pth")
        torch.save(rc_model.state_dict(), ckpt_path)

    def _log_epoch_summary(
        self,
        epoch: int,
        phase: str,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._log_header(f"{phase} {epoch + 1}/{self.args.max_epochs}")
        print(f"train_loss          : {train_loss:.6f}")
        print(f"val_loss            : {val_loss:.6f}")
        print(f"best_val_loss       : {best_val_loss:.6f}")
        if extra:
            for key, value in extra.items():
                print(f"{key:<20}: {value}")

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _teacher_stats(self, student_pred: torch.Tensor, teacher_pred: torch.Tensor, target: torch.Tensor):
        """Build the teacher error terms used in the paper-defined state."""
        teacher_gt_error = ((teacher_pred - target) ** 2).mean(dim=1)
        teacher_student_gap = ((teacher_pred - student_pred) ** 2).mean(dim=1)
        return teacher_gt_error, teacher_student_gap

    def _samplewise_teacher_losses(self, student_pred: torch.Tensor, teacher_pred: torch.Tensor):
        """Return per-sample, per-teacher KD losses before weighting."""
        per_teacher = []
        for teacher_idx in range(teacher_pred.shape[-1]):
            loss = ((student_pred - teacher_pred[..., teacher_idx : teacher_idx + 1]) ** 2).mean(dim=(1, 2))
            per_teacher.append(loss)
        return torch.stack(per_teacher, dim=-1)

    def _current_sample_error(self, student_pred: torch.Tensor, target: torch.Tensor):
        return ((student_pred - target) ** 2).mean(dim=(1, 2))

    def _validation_error(self, rc_model, val_loader):
        """Compute the held-out validation loss used in the reward term."""
        rc_model.eval()
        losses = []
        with torch.no_grad():
            for _, _, _, _, batch_x_m, batch_y_m, _ in val_loader:
                batch_x_m = batch_x_m.float().to(self.args.device)
                batch_y_m = batch_y_m.float().to(self.args.device)
                pred = rc_model.multistep_predict(batch_x_m)
                losses.append(self.criterion(pred, batch_y_m))
        rc_model.train()
        return torch.stack(losses).mean()

    def _physical_consistency_regularization(self, rc_model):
        """Penalize RC time constants that become too small relative to the sampling interval."""
        dt = torch.tensor(self.args.delta_t, dtype=torch.float32, device=self.args.device)
        ratios = []
        if hasattr(rc_model, "R") and hasattr(rc_model, "C"):
            ratios.append(dt / (rc_model.R * rc_model.C))
        if hasattr(rc_model, "Ri") and hasattr(rc_model, "C"):
            ratios.append(dt / (rc_model.Ri * rc_model.C))
        if hasattr(rc_model, "Re") and hasattr(rc_model, "C"):
            ratios.append(dt / (rc_model.Re * rc_model.C))
        if hasattr(rc_model, "Ri") and hasattr(rc_model, "Ci"):
            ratios.append(dt / (rc_model.Ri * rc_model.Ci))
        if hasattr(rc_model, "Ri") and hasattr(rc_model, "Ce"):
            ratios.append(dt / (rc_model.Ri * rc_model.Ce))
        if hasattr(rc_model, "Re") and hasattr(rc_model, "Ce"):
            ratios.append(dt / (rc_model.Re * rc_model.Ce))
        if not ratios:
            return torch.tensor(0.0, device=self.args.device)
        return torch.stack([ratio.pow(2) for ratio in ratios]).mean()

    def _run_name(self) -> str:
        stem = Path(self.args.file_name).stem
        if stem.startswith("house_id_"):
            stem = stem[len("house_id_") :]
        return stem[:24]

    def save_metrics_to_json(self, metrics):
        import json

        output_dir = os.path.join(self.args.project_root, self.args.result_dir, self.args.exp_name)
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{self.args.rc_model}_{self._run_name()}.json")
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump({"state": self.args.state_dataset, "metrics": metrics}, file, indent=4)

    def val(self, rc_model, _, val_loader):
        rc_model.eval()
        losses = []
        with torch.no_grad():
            for _, _, _, _, batch_x_m, batch_y_m, _ in val_loader:
                batch_x_m = batch_x_m.float().to(self.args.device)
                batch_y_m = batch_y_m.float().to(self.args.device)
                pred = rc_model.multistep_predict(batch_x_m)
                losses.append(self.criterion(pred, batch_y_m))
        rc_model.train()
        return torch.stack(losses).mean()

    def test(self, rc_model, _, test_loader):
        truth_s, pred_s, truth_m, pred_m = [], [], [], []
        rc_model.eval()
        with torch.no_grad():
            for _, batch_x_s, batch_y_s, _, batch_x_m, batch_y_m, _ in test_loader:
                batch_x_s = batch_x_s.float().to(self.args.device)
                batch_y_s = batch_y_s.float().to(self.args.device)
                batch_x_m = batch_x_m.float().to(self.args.device)
                batch_y_m = batch_y_m.float().to(self.args.device)

                truth_s.append(batch_y_s.detach().cpu().numpy().squeeze())
                truth_m.append(batch_y_m.detach().cpu().numpy().squeeze())
                pred_s.append(rc_model.onestep_predict(batch_x_s).detach().cpu().numpy().squeeze())
                pred_m.append(rc_model.multistep_predict(batch_x_m).detach().cpu().numpy().squeeze())

        truth_s = np.concatenate(truth_s)
        truth_m = np.concatenate(truth_m)
        pred_s = np.concatenate(pred_s)
        pred_m = np.concatenate(pred_m)
        rc_model.train()
        return (
            mae(truth_s, pred_s),
            mse(truth_s, pred_s),
            rmse(truth_s, pred_s),
            cv_rmse(truth_s, pred_s),
            mae(truth_m, pred_m),
            mse(truth_m, pred_m),
            rmse(truth_m, pred_m),
            cv_rmse(truth_m, pred_m),
        )

    def plot_test_results(self, rc_model, _, test_loader):
        rc_model.eval()
        with torch.no_grad():
            plt.figure(figsize=(12, 6))
            y_pred_s, y_s = [], []
            for _, batch_x_s, batch_y_s, _, _, _, _ in test_loader:
                batch_x_s = batch_x_s.float().to(self.args.device)
                batch_y_s = batch_y_s.float().to(self.args.device)
                y_pred_s.append(rc_model.onestep_predict(batch_x_s).detach().cpu().numpy().squeeze())
                y_s.append(batch_y_s.detach().cpu().numpy().squeeze())
            y_pred_s = np.concatenate(y_pred_s)
            y_s = np.concatenate(y_s)

            plt.subplot(2, 1, 1)
            plt.plot(y_s, color="grey", label="Actual")
            plt.plot(y_pred_s, color="blue", label="Predicted")
            plt.legend()

            y_pred_m, y_m = [], []
            for _, _, _, _, batch_x_m, batch_y_m, _ in test_loader:
                batch_x_m = batch_x_m.float().to(self.args.device)
                batch_y_m = batch_y_m.float().to(self.args.device)
                y_pred_m.append(rc_model.multistep_predict(batch_x_m).detach().cpu().numpy().squeeze())
                y_m.append(batch_y_m.detach().cpu().numpy().squeeze())
            y_pred_m = np.stack(y_pred_m)
            y_m = np.stack(y_m)
            if y_pred_m.ndim < 3:
                y_pred_m = np.expand_dims(y_pred_m, axis=-1)
            n_windows, batch_size, horizon = y_pred_m.shape
            valid_idx = [idx * horizon for idx in range((n_windows * batch_size) // horizon)]
            y_pred_m = y_pred_m.reshape(n_windows * batch_size, horizon)[valid_idx, :].flatten()
            y_m = y_m.reshape(n_windows * batch_size, horizon)[valid_idx, :].flatten()

            plt.subplot(2, 1, 2)
            plt.plot(y_m, color="grey", label="Actual")
            plt.plot(y_pred_m, color="blue", label="Predicted")
            plt.legend()

            output_dir = os.path.join(self.args.project_root, self.args.graph_dir, self.args.exp_name)
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f"{self.args.rc_model}_{self._run_name()}.svg")
            plt.savefig(fig_path)
            plt.close()
        rc_model.train()

    def _run_episode(self, rc_model, actor, critic, rc_optim, train_loader, val_loader):
        """Run one training episode and collect PPO rollout data."""
        buffer = EpisodeBuffer([], [], [], [], [], [], [], [])
        total_train_loss = 0.0

        for batch_x_ctx, _, _, _, batch_x_m, batch_y_m, batch_y_tsfm_m in train_loader:
            batch_x_ctx = batch_x_ctx.float().to(self.args.device)
            batch_x_m = batch_x_m.float().to(self.args.device)
            batch_y_m = batch_y_m.float().to(self.args.device)
            batch_y_tsfm_m = batch_y_tsfm_m.float().to(self.args.device)

            rc_model.eval()
            with torch.no_grad():
                student_pred_old = rc_model.multistep_predict(batch_x_m)
                teacher_gt_error, teacher_student_gap = self._teacher_stats(student_pred_old, batch_y_tsfm_m, batch_y_m)

            alpha, beta = actor(batch_x_ctx, teacher_gt_error, teacher_student_gap)
            dist = Beta(alpha, beta)
            actions = dist.rsample()
            log_prob = dist.log_prob(actions).sum(dim=-1)
            value = critic(batch_x_ctx, teacher_gt_error, teacher_student_gap)

            rc_model.train()
            rc_optim.zero_grad()
            student_pred = rc_model.multistep_predict(batch_x_m)
            gt_loss = self.criterion(student_pred, batch_y_m)
            distill_loss_matrix = self._samplewise_teacher_losses(student_pred, batch_y_tsfm_m)
            # The sampled Beta weights define the per-sample teacher mixture for MTKD.
            weighted_distill_loss = (actions * distill_loss_matrix).sum(dim=-1).mean()
            phy_loss = self._physical_consistency_regularization(rc_model)
            total_loss = gt_loss + self.args.alpha_m * weighted_distill_loss + self.args.phy_reg_weight * phy_loss
            total_loss.backward()
            rc_optim.step()
            total_train_loss += total_loss.item()

            with torch.no_grad():
                updated_pred = rc_model.multistep_predict(batch_x_m)
                sample_error = self._current_sample_error(updated_pred, batch_y_m)
                validation_error = self._validation_error(rc_model, val_loader)
                reward = -(self.args.reward_eta * sample_error + (1.0 - self.args.reward_eta) * validation_error)

            buffer.batch_x.append(batch_x_ctx.detach())
            buffer.teacher_gt_error.append(teacher_gt_error.detach())
            buffer.teacher_student_gap.append(teacher_student_gap.detach())
            buffer.actions.append(actions.detach())
            buffer.log_probs.append(log_prob.detach())
            buffer.values.append(value.detach())
            buffer.rewards.append(reward.detach())
            buffer.masks.append(torch.ones_like(reward.detach()))

        mean_reward = torch.stack(buffer.rewards).mean().item()
        action_summary = self._action_summary(torch.cat(buffer.actions, dim=0))
        return buffer, total_train_loss, mean_reward, action_summary

    def _update_policy(self, actor, critic, actor_optim, critic_optim, buffer: EpisodeBuffer):
        """Update actor and critic using PPO over the collected episode rollout."""
        with torch.no_grad():
            last_value = critic(buffer.batch_x[-1], buffer.teacher_gt_error[-1], buffer.teacher_student_gap[-1])

        values = torch.stack(buffer.values + [last_value], dim=0)
        rewards = torch.stack(buffer.rewards, dim=0)
        masks = torch.stack(buffer.masks, dim=0)
        old_log_probs = torch.stack(buffer.log_probs, dim=0)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            masks=masks,
            gamma=self.args.gamma,
            lam=self.args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actions = torch.stack(buffer.actions, dim=0)
        batch_x = torch.stack(buffer.batch_x, dim=0)
        teacher_gt_error = torch.stack(buffer.teacher_gt_error, dim=0)
        teacher_student_gap = torch.stack(buffer.teacher_student_gap, dim=0)

        ppo_stats = []
        for ppo_idx in range(self.args.max_ppo_epochs):
            new_log_probs = []
            entropies = []
            new_values = []
            for idx in range(batch_x.shape[0]):
                alpha, beta = actor(batch_x[idx], teacher_gt_error[idx], teacher_student_gap[idx])
                dist = Beta(alpha, beta)
                new_log_probs.append(dist.log_prob(actions[idx]).sum(dim=-1))
                entropies.append(dist.entropy().sum(dim=-1))
                new_values.append(critic(batch_x[idx], teacher_gt_error[idx], teacher_student_gap[idx]))

            new_log_probs = torch.stack(new_log_probs, dim=0)
            entropies = torch.stack(entropies, dim=0)
            new_values = torch.stack(new_values, dim=0)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.args.ppo_clip_eps, 1.0 + self.args.ppo_clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.args.entropy_coef * entropies.mean()
            critic_loss = self.criterion(new_values, returns)

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            ppo_stats.append(
                {
                    "ppo_epoch": ppo_idx + 1,
                    "policy_loss": float(actor_loss.detach().cpu()),
                    "value_loss": float(critic_loss.detach().cpu()),
                }
            )
        return ppo_stats

    def train(self):
        train_data, train_loader = self._get_data("train")
        val_data, val_loader = self._get_data("val")
        test_data, test_loader = self._get_data("test")

        rc_model = select_rc_model(self.args).to(self.args.device)
        actor = ThermoStillActor(self.args).to(self.args.device)
        critic = ThermoStillCritic(self.args).to(self.args.device)
        rc_optim = torch.optim.Adam(rc_model.parameters(), lr=self.args.rc_lr)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.args.policy_lr)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=self.args.value_lr)
        best_state = copy.deepcopy(rc_model.state_dict())
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.args.max_epochs):
            if epoch < self.args.pretrain_epochs:
                pretrain_loss = 0.0
                for _, _, _, _, batch_x_m, batch_y_m, _ in train_loader:
                    batch_x_m = batch_x_m.float().to(self.args.device)
                    batch_y_m = batch_y_m.float().to(self.args.device)
                    rc_optim.zero_grad()
                    pred = rc_model.multistep_predict(batch_x_m)
                    loss = self.criterion(pred, batch_y_m)
                    loss.backward()
                    rc_optim.step()
                    pretrain_loss += loss.item()
                val_loss = float(self.val(rc_model, val_data, val_loader))
                current_best = min(best_val_loss, val_loss)
                self._log_epoch_summary(
                    epoch=epoch,
                    phase="Pretrain Epoch",
                    train_loss=pretrain_loss,
                    val_loss=val_loss,
                    best_val_loss=current_best,
                )
            else:
                buffer, total_train_loss, mean_reward, action_summary = self._run_episode(
                    rc_model, actor, critic, rc_optim, train_loader, val_loader
                )
                ppo_stats = self._update_policy(actor, critic, actor_optim, critic_optim, buffer)
                val_loss = float(self.val(rc_model, val_data, val_loader))
                extra = {
                    "mean_reward": f"{mean_reward:.6f}",
                    "teacher_weights": action_summary,
                }
                for stat in ppo_stats:
                    extra[f"ppo_{stat['ppo_epoch']}"] = (
                        f"policy_loss={stat['policy_loss']:.6f}, value_loss={stat['value_loss']:.6f}"
                    )
                self._log_epoch_summary(
                    epoch=epoch,
                    phase="Episode",
                    train_loss=total_train_loss,
                    val_loss=val_loss,
                    best_val_loss=min(best_val_loss, val_loss),
                    extra=extra,
                )

            if val_loss < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
                best_val_loss = val_loss
                best_state = copy.deepcopy(rc_model.state_dict())
                self._maybe_save_checkpoint(rc_model)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"EarlyStopping counter: {patience_counter} out of {self.args.patience}")
                if patience_counter >= self.args.patience:
                    print("Early stopping")
                    break

            adjust_learning_rate(rc_optim, epoch, self.args.rc_lr, self.args.max_epochs, self.args.lradj)
            adjust_learning_rate(actor_optim, epoch, self.args.policy_lr, self.args.max_epochs, self.args.lradj)
            adjust_learning_rate(critic_optim, epoch, self.args.value_lr, self.args.max_epochs, self.args.lradj)

        rc_model.load_state_dict(best_state)
        self.plot_test_results(rc_model, test_data, test_loader)
        test_metrics = self.test(rc_model, test_data, test_loader)

        self._log_header("Final Summary")
        print(f"best model metrics   : {self._format_metrics_line(test_metrics)}")
        rc_model.describe_model()

        metrics = {
            "one_step": {
                "mae": float(test_metrics[0]),
                "mse": float(test_metrics[1]),
                "rmse": float(test_metrics[2]),
                "cv_rmse": float(test_metrics[3]),
            },
            "multi_step": {
                "mae": float(test_metrics[4]),
                "mse": float(test_metrics[5]),
                "rmse": float(test_metrics[6]),
                "cv_rmse": float(test_metrics[7]),
            },
        }
        self.save_metrics_to_json(metrics)
        return test_metrics
