"""
train_tan_dmc.py -- Deep Monte-Carlo (DouZero-style) self-play on Tan via RLCard.
    python train_tan_dmc.py --num_actors 8 --total_frames 20000000              # CPU
    python train_tan_dmc.py --cuda 0 --training_device 0                         # GPU 0
    python train_tan_dmc.py --eval_only                                          # evaluate latest checkpoint
"""
import os
import re
import glob
import time
import threading
import argparse
import multiprocessing as mp
import torch

from rlcard.utils import set_seed

from tan_common import add_game_args, make_env, evaluate_vs_random, load_torch_agent

MILESTONE_FRAMES = 100_000_000   # permanent snapshot kept every 100M frames
JANITOR_PERIOD_SECS = 30         # how often to sweep the save dir for cleanup

# --- Workaround: skip optimizer load when state is None ---
_orig_load = torch.optim.Optimizer.load_state_dict
def _patched_load(self, state_dict):
    if state_dict is None:
        return
    return _orig_load(self, state_dict)
torch.optim.Optimizer.load_state_dict = _patched_load
# --------------------------------------------------------


def _frames_of(path):
    m = re.search(r"_(\d+)\.pth$", path)
    return int(m.group(1)) if m else -1


def latest_checkpoint(run_dir, seat):
    paths = glob.glob(os.path.join(run_dir, f"{seat}_*.pth"))   # DMC saves '<seat>_<frames>.pth'
    if not paths:
        return None
    return max(paths, key=_frames_of)


def prune_checkpoints(run_dir, num_seats):
    """
    Retention policy:
      - keep the single most-recent checkpoint per seat (rolling "latest")
      - keep one permanent milestone checkpoint per seat for every
        100M-frame bucket that has been reached
      - delete everything else
    """
    for seat in range(num_seats):
        paths = glob.glob(os.path.join(run_dir, f"{seat}_*.pth"))
        if not paths:
            continue

        by_frames = sorted(((p, _frames_of(p)) for p in paths), key=lambda x: x[1])

        keep = set()
        last_bucket = -1
        for p, f in by_frames:
            bucket = f // MILESTONE_FRAMES
            if bucket > last_bucket:        # first file to reach this 100M bucket -> permanent
                keep.add(p)
                last_bucket = bucket

        newest_path, _ = by_frames[-1]      # always keep the newest as the rolling "latest"
        keep.add(newest_path)

        for p, _ in by_frames:
            if p not in keep:
                try:
                    os.remove(p)
                except OSError:
                    pass


def janitor_loop(run_dir, num_seats, stop_event, period=JANITOR_PERIOD_SECS):
    while not stop_event.is_set():
        prune_checkpoints(run_dir, num_seats)
        stop_event.wait(period)


def train(args):
    from rlcard.agents.dmc_agent import DMCTrainer
    set_seed(args.seed)
    env = make_env(args)

    use_mps = args.mps and torch.backends.mps.is_available()
    trainer = DMCTrainer(
        env,
        cuda="" if use_mps else args.cuda,      # "" -> actors + buffers on CPU
        load_model=args.load_model,
        xpid=args.xpid,
        savedir=args.savedir,
        save_interval=args.save_interval,
        num_actor_devices=args.num_actor_devices,
        num_actors=args.num_actors,
        training_device=args.training_device if args.cuda else "cpu",
        total_frames=args.total_frames,
        exp_epsilon=args.exp_epsilon,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        learning_rate=args.learning_rate,
    )
    if use_mps:
        trainer.training_device = "mps"         # override after __init__ forces "cpu"
        print("learner on MPS, actors on CPU")

    run_dir = os.path.join(args.savedir, args.xpid)
    stop_event = threading.Event()
    janitor = threading.Thread(
        target=janitor_loop,
        args=(run_dir, env.num_players, stop_event),
        daemon=True,
    )
    janitor.start()

    try:
        print(f"[debug] trainer.frames before start() = {getattr(trainer, 'frames', 'MISSING')}")
        print(f"[debug] trainer.load_model       = {trainer.load_model}")
        print(f"[debug] trainer.total_frames     = {trainer.total_frames}")
        trainer.start()                             # blocks until total_frames is reached
    except KeyboardInterrupt:
        print("\ninterrupted -- evaluating latest checkpoint")
    finally:
        stop_event.set()
        janitor.join(timeout=5)
        prune_checkpoints(run_dir, env.num_players)  # final sweep
        for p in mp.active_children():             # actor processes run forever; stop them
            p.terminate()


def evaluate(args):
    run_dir = os.path.join(args.savedir, args.xpid)
    eval_env = make_env(args, seed_offset=1)
    for seat in range(eval_env.num_players):
        ckpt = latest_checkpoint(run_dir, seat)
        if ckpt is None:
            print(f"seat {seat}: no checkpoint in {run_dir}")
            continue
        agent = load_torch_agent(ckpt, "cpu")
        score = evaluate_vs_random(eval_env, agent, args.num_eval_games, seat=seat)
        print(f"seat {seat}: {os.path.basename(ckpt)}  avg payoff vs random = {score:+.4f}")


if __name__ == "__main__":
    p = add_game_args(argparse.ArgumentParser("DMC on Tan"))
    p.add_argument("--cuda", type=str, default="", help="'' for CPU, e.g. '0' or '0,1' for GPUs")
    p.add_argument("--training_device", type=str, default="0")
    p.add_argument("--num_actor_devices", type=int, default=1)
    p.add_argument("--num_actors", type=int, default=5)
    p.add_argument("--total_frames", type=int, default=20_000_000)
    p.add_argument("--exp_epsilon", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--unroll_length", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--save_interval", type=int, default=5,
                    help="minutes between DMC checkpoint writes (tune so this roughly "
                         "matches ~10M frames at your actors' throughput)")
    p.add_argument("--load_model", action="store_true")
    p.add_argument("--xpid", type=str, default="tan")
    p.add_argument("--savedir", type=str, default="experiments/tan_dmc")
    p.add_argument("--num_eval_games", type=int, default=500)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--mps", action="store_true", help="train the learner on Apple Metal (MPS)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda   # must be set before CUDA initialises
    if not args.eval_only:
        train(args)
    evaluate(args)