import math
import warnings
from torch.optim import Optimizer
import matplotlib.pyplot as plt


class Scheduler:
    def __init__(self, torch_optimizer: Optimizer, schedule: str, training_steps: int, warmup_steps: int, max_lr: float, min_lr: float):
        """
        A custom learning rate scheduler class for this project.
        Do note that warmup and min_lr will be ignored if using "constant" schedule for obvious reasons,
            just about any arbitrary value for those two are fine as placeholders in that case
        Warmup schedule would be fixed at linear, after that, it will be either "linear" or "cosine" as defined by user

        :param schedule: Type of schedule to use. Options are ["cosine", "linear", "constant"]. Constant schedule uses max lr. Then again, a scheduler isn't even needed lol
        :param training_steps: Number of total training steps (including warmup)
        :param warmup_steps: Number of steps for the model to "warmup"
        :param max_lr: Maximum (base) learning rate
        :param min_lr: Minimum learning rate
        """

        # Basics checks, should put realistic values. Too extreme (small) for # steps might result in unexpected behavior
        assert schedule in ["cosine", "linear", "constant"], "Invalid scheduling method"
        assert training_steps > 0
        assert warmup_steps <= training_steps
        assert max_lr > 0
        assert 0 < min_lr <= max_lr


        self.optimizer = torch_optimizer
        self.schedule = schedule
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = training_steps - warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.current_lr = None
        self.warned = False

        # These two are used for checkpointing
        self.cur_scheduler_steps = 0
        self.prev_scheduler_steps = 0


        # One time set
        if schedule == "constant":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max_lr


    def _update_cosine(self, current_step: int):
        current_step -= self.warmup_steps  # Offset by warmup steps

        scale = (current_step / self.decay_steps) * 3.14159  # Get range of [1, 0] for cosine decay

        # Couldn't quite get the tail right, asked GPT after multiple attempts lol
        # This works, but tail would be approaching 0 rather than min_lr, which overall doesn't really matter since it's just the last few steps
        # But...better to stick to min_lr I suppose
        # lr = (self.max_lr + (math.cos(scale) * self.max_lr)) / 2

        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(scale))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


    def _update_linear(self, current_step: int):
        current_step -= self.warmup_steps  # Offset by warmup steps

        lr = self.max_lr - (current_step / self.decay_steps) * (self.max_lr - self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


    def _update_warmup(self, current_step: int):
        # Have lr of step==0 to be lr of step==1 (Otherwise lr would be 0 and that's meaningless)
        lr = (max(1, current_step) / self.warmup_steps) * self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


    def step(self, current_step: int):
        current_step += self.prev_scheduler_steps  # Account for checkpointing, if used
        self.cur_scheduler_steps += 1  # Increment number of steps

        # Check if over-stepped
        if current_step > self.training_steps and not self.warned:
            warnings.warn(f"Current step ({current_step}) has overstepped set training_steps ({self.training_steps}), now using min_lr ({self.min_lr})")
            self.warned = True

        # Check if in warmup phase
        if current_step < self.warmup_steps and self.schedule != "constant":
            self.current_lr = self._update_warmup(current_step)
            return

        if self.schedule == "cosine":
            # Torch's cosine decay trends back upwards if overstepping T_max, not sure why
            # For this implementation, it will stay at lower bound if overstepped
            self.current_lr = self._update_cosine(current_step) if not self.warned else self.min_lr
        elif self.schedule == "linear":
            self.current_lr = self._update_linear(current_step) if not self.warned else self.min_lr
        elif self.schedule == "constant":
            self.current_lr = self.max_lr
        else:
            raise ValueError("This shouldn't happen")


    def get_current_lr(self):
        return self.current_lr


    def state_dict(self):
        return {
            "schedule": self.schedule,
            "training_steps": self.training_steps,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "current_lr": self.current_lr,
            "warned": self.warned,
            "prev_scheduler_steps": self.prev_scheduler_steps + self.cur_scheduler_steps
        }


    def load_state_dict(self, config: dict, strict=True):
        if strict:
            for key in ("schedule", "training_steps", "warmup_steps", "max_lr", "min_lr"):
                assert getattr(self, key) == config[key], f"Mismatch in {key}"
        else:
            for key in ("schedule", "training_steps", "warmup_steps", "max_lr", "min_lr"):
                setattr(self, key, config[key])

        self.decay_steps = config["training_steps"] - config["warmup_steps"]
        self.current_lr = config["current_lr"]
        self.prev_scheduler_steps = config["prev_scheduler_steps"]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        # Warn again?
        # self.warned = config["warned"]


if __name__ == "__main__":
    # Testing to make sure things are right
    class FakeOptimizer:
        def __init__(self):
            self.param_groups = []


    fake_optim = FakeOptimizer()

    total_steps = 100_000
    lrs = []
    scheduler = Scheduler(fake_optim, "cosine",
                          training_steps=total_steps,
                          warmup_steps=1000000000,
                          max_lr=6e-4,
                          min_lr=9e-5)

    for i in range(total_steps + 10_000):  # Testing overstep by 10k
        scheduler.step(i)
        lrs.append(scheduler.get_current_lr())

    print(lrs[:100])
    print(lrs[-100:])
    plt.plot(lrs)
    plt.show()


