
# learning rate schedule: stable then decay
def get_lr(step, num_step, cooldown_frac=0.4):
    x = step / num_step  # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac
        # return w * 1.0 + (1 - w) * 0.1


def get_lr_trapezoidal(step, num_step, warmup_iters=256, cooldown_frac=0.4):
    """Trapezoidal schedule from modded-nanogpt 844e5fd.

    Linear warmup, constant plateau, then linear warmdown to zero.
    Returns a multiplier in [0, 1] applied to the base learning rate.
    """
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    return get_lr(step, num_step, cooldown_frac)
