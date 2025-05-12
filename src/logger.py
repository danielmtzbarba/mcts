import os
import sys
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter

file_handler = logging.FileHandler(filename="experiment.log")  # , mode="w")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    handlers=handlers,
    format="[%(asctime)s] %(levelname)s ==> %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    encoding="utf-8",
    level=logging.INFO,
)
logger = logging.getLogger("experiment")


class DRLogger(object):
    def __init__(self, run_name) -> None:
        self._run_name = run_name
        self._logger = logger
        log_dir = os.path.join("runs", run_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self._num_ep = 1

    def episode(self, info, average_reward):
        # Console logging
        print(
            f"[Self-Play] Episode {self._num_ep}: Return = {info["ep"]["return"][0]:.2f}"
        )

        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar(
                "Stats/episode_return",
                info["termination"]["return"][0],
                self._num_ep,
            )
            self.writer.add_scalar(
                "Stats/episode_length",
                info["termination"]["length"][0],
                self._num_ep,
            )
            self.writer.add_scalar(
                "Stats/distance2target",
                info["env"]["dist2target_t"][0],
                self._num_ep,
            )

            if self._num_ep % 10 == 0:  # Log moving average every 10 iterations
                self.writer.add_scalar(
                    "Stats/moving_avg_rwd",
                    average_reward,
                    self._num_ep,
                )
        self._num_ep += 1
        return

    def losses(self, total, mean, value, policy, reward):
        # Console logging
        print(f"[Train] Step {self._num_ep}: Total Loss = {total:.4f}")

        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar("Loss/total", mean, self._num_ep)
            self.writer.add_scalar("Loss/value", value, self._num_ep)
            self.writer.add_scalar("Loss/policy", policy, self._num_ep)
            self.writer.add_scalar("Loss/reward", reward, self._num_ep)
        return

    def evaluation(self, rets, lens):
        rets = np.array(rets)
        lens = np.array(lens)
        # Console logging
        print("\n--- Evaluation ---")
        for i, ret in enumerate(rets):
            print(f"[Evaluation]-{i}: {ret:.4f}")
        print(
            f"[Evaluation]: Return -> Mean={np.mean(rets):.4f}, STD={np.std(rets):.4f}, Var={np.var(rets):.4f}"
        )

        if self.writer:
            self.writer.add_scalar(
                "Eval/eval_mean_return",
                np.mean(rets),
                self._num_ep,
            )
            self.writer.add_scalar(
                "Eval/eval_std_return",
                np.std(rets),
                self._num_ep,
            )
            self.writer.add_scalar(
                "Eval/eval_var_return",
                np.var(rets),
                self._num_ep,
            )

    @property
    def num_ep(self):
        return self._num_ep

    @property
    def logger(self):
        return self._logger
