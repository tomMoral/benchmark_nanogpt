"""Analysis helpers for the CIFAR-10 / ResNet-18 benchmark results.

Reusable replacement for the throwaway scripts used while tuning. Importable
(use the functions on a dataframe) or runnable from the CLI:

    python scripts/investigate_cifar10.py                       # leaderboard
    python scripts/investigate_cifar10.py -f outputs/cifar10.parquet
    python scripts/investigate_cifar10.py sweep Muon
    python scripts/investigate_cifar10.py traj Scion --lr 0.002
    python scripts/investigate_cifar10.py plateau Muon --lr 0.01

`objective_value` is the top-1 test error (the ResNet's eval metric); `time` is
training-only (benchopt pauses the timer during evaluation).
"""
import argparse

import numpy as np
import pandas as pd
from benchopt.results import read_results

DEFAULT_FILE = "outputs/cifar10_merged.parquet"
# Historical hardcoded cooldown fractions, used only for runs predating the
# cooldown_frac parameter (no p_solver_cooldown_frac column).
LEGACY_COOLDOWN = {"Muon": 0.29, "Scion": 0.28}


def load(path=DEFAULT_FILE):
    """Read a results parquet and add a bare `solver` (family) column."""
    df = read_results(path)
    df["solver"] = df["solver_name"].str.split("[").str[0]
    return df


def lr_of(row):
    """Learning rate of a run, whatever the solver calls it."""
    for col in ("p_solver_muon_lr", "p_solver_learning_rate"):
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return np.nan


def cooldown_of(row):
    col = "p_solver_cooldown_frac"
    if col in row and pd.notna(row[col]):
        return float(row[col])
    return LEGACY_COOLDOWN.get(row["solver"], np.nan)


def _curve(df, solver_name):
    return df[df["solver_name"] == solver_name].sort_values("stop_val")


def select(df, solver=None, lr=None):
    """Return the list of solver_name strings matching a family / lr filter."""
    g = df if solver is None else df[df["solver"] == solver]
    names = []
    for name in g["solver_name"].unique():
        row = _curve(df, name).iloc[-1]
        if lr is None or np.isclose(lr_of(row), lr):
            names.append(name)
    return names


def leaderboard(df):
    """One row per run: best/final test error, schedule, training time."""
    rows = []
    final = (df.sort_values("stop_val")
               .groupby("solver_name", as_index=False).last())
    for _, r in final.iterrows():
        g = _curve(df, r["solver_name"])
        err = g["objective_value"].values * 100
        rows.append(dict(
            solver=r["solver"],
            lr=lr_of(r),
            num_steps=int(r["p_solver_num_steps"]),
            cooldown=cooldown_of(r),
            steps_run=int(r["stop_val"]),
            best_err=round(float(np.nanmin(err)), 2),
            final_err=round(float(err[-1]), 2),
            train_loss=float(r["objective_train_loss"]),
            time_s=round(float(g["time"].max()), 1),
        ))
    return (pd.DataFrame(rows)
              .sort_values("best_err")
              .reset_index(drop=True))


def sweep(df, solver):
    """Leaderboard rows for one solver family, ordered by lr/schedule."""
    board = leaderboard(df)
    return (board[board["solver"] == solver]
            .sort_values(["lr", "num_steps", "cooldown"])
            .reset_index(drop=True))


def grid(df, solver, value="best_err"):
    """Pivot `value` over the (num_steps x cooldown_frac) schedule grid."""
    board = leaderboard(df)
    sub = board[board["solver"] == solver]
    return sub.pivot_table(index="num_steps", columns="cooldown",
                           values=value, aggfunc="min")


def trajectory(df, solver, lr=None, n=13):
    """Print a sparse step / test-error / train-loss trajectory per run."""
    for name in select(df, solver, lr):
        g = _curve(df, name)
        err = g["objective_value"].values * 100
        tl = g["objective_train_loss"].values
        steps = g["stop_val"].values.astype(int)
        idx = np.linspace(0, len(steps) - 1, min(n, len(steps))).astype(int)
        print(f"=== {name} ===")
        print(f"  min {err.min():.2f}% @ {steps[int(np.argmin(err))]} | "
              f"final {err[-1]:.2f}% @ {steps[-1]}")
        print("  " + "  ".join(f"{steps[i]}:{err[i]:.1f}%(tl{tl[i]:.2g})"
                               for i in idx))
        print()


def plateau(df, solver=None, lr=None):
    """Pre-cooldown (flat-phase) plateau vs the cooldown's contribution.

    Answers "could the cooldown start earlier?": if the flat phase plateaus
    well before cooldown_onset, the tail of the flat phase is wasted.
    """
    rows = []
    for name in select(df, solver, lr):
        g = _curve(df, name)
        r = g.iloc[-1]
        err = g["objective_value"].values * 100
        steps = g["stop_val"].values.astype(int)
        num = int(r["p_solver_num_steps"])
        cd_onset = round((1 - cooldown_of(r)) * num)

        flat = steps <= cd_onset
        ferr, fsteps = err[flat], steps[flat]
        fmin = ferr.min()
        # plateau onset: first flat step within 0.3pp of the flat-phase min
        ip = int(np.where(ferr <= fmin + 0.3)[0][0])
        rows.append(dict(
            solver=r["solver"], lr=lr_of(r), num_steps=num,
            cooldown=cooldown_of(r), cooldown_onset=cd_onset,
            flat_min=round(fmin, 2), plateau_step=int(fsteps[ip]),
            precooldown_err=round(float(ferr[-1]), 2),
            final_err=round(float(err[-1]), 2),
            cooldown_drop=round(float(ferr[-1] - err[-1]), 2),
        ))
    return pd.DataFrame(rows)


def _main():
    pd.set_option("display.width", 260)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-f", "--file", default=DEFAULT_FILE)
    sub = p.add_subparsers(dest="cmd")
    for cmd in ("sweep", "traj", "plateau", "grid"):
        sp = sub.add_parser(cmd)
        sp.add_argument("solver", nargs="?" if cmd == "plateau" else None)
        sp.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    df = load(args.file)
    if args.cmd == "sweep":
        print(sweep(df, args.solver).to_string(index=False))
    elif args.cmd == "grid":
        print(grid(df, args.solver).to_string())
    elif args.cmd == "traj":
        trajectory(df, args.solver, args.lr)
    elif args.cmd == "plateau":
        print(plateau(df, args.solver, args.lr).to_string(index=False))
    else:
        print(leaderboard(df).to_string(index=False))


if __name__ == "__main__":
    _main()
