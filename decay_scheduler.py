class ExponentialDecayScheduler:
    """Exact match for TensorFlow ExponentialDecay with staircase=True."""
    def __init__(self, initial_lr, decay_steps, decay_rate, staircase=True):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def get_lr(self, step):
        if self.staircase:
            exponent = step // self.decay_steps
        else:
            exponent = step / self.decay_steps
        return self.initial_lr * (self.decay_rate ** exponent)