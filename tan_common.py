"""
tan_common.py -- helpers shared by the Tan training scripts.
"""
import copy

import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import tournament

import tan_rlcard  # noqa: F401  -- importing registers the 'tan' environment


def add_game_args(p):
    p.add_argument("--num_players", type=int, default=3)
    p.add_argument("--hand_size", type=int, default=6)
    p.add_argument("--attacker_order", choices=["main_first", "round_robin", "random"],
                   default="main_first")
    p.add_argument("--seed", type=int, default=42)
    return p


def game_config(args):
    return {
        "game_num_players": args.num_players,
        "game_hand_size": args.hand_size,
        "game_defender_first": True,
        "game_attacker_order": args.attacker_order,
    }


def make_env(args, seed_offset=0, **extra_config):
    return rlcard.make("tan", config={"seed": args.seed + seed_offset,
                                      **game_config(args), **extra_config})


def random_agents(env, n):
    return [RandomAgent(num_actions=env.num_actions) for _ in range(n)]


def evaluate_vs_random(eval_env, agent, num_games, seat=0):
    """Average payoff of `agent` in `seat` with random agents in every other seat (uses eval_step)."""
    agents = random_agents(eval_env, eval_env.num_players)
    agents[seat] = agent
    eval_env.set_agents(agents)
    return tournament(eval_env, num_games)[seat]


def load_torch_agent(path, device="cpu"):
    """Load an agent saved with torch.save(agent, ...).
    torch>=2.6 defaults to weights_only=True, which refuses whole pickled agents."""
    try:
        agent = torch.load(path, map_location=device, weights_only=False)
    except TypeError:                      # older torch without the kwarg
        agent = torch.load(path, map_location=device)
    if hasattr(agent, "set_device"):       # DQN / NFSP / DMC agents all have this
        agent.set_device(device)
    return agent


def install_snapshot_step_back(game):
    """Fallback for games that don't implement step_back(): deep-copy the game's
    state before every step and restore it on step_back. Slow, but generic.
    Restoring also rewinds the game's RNG, so stepping again repeats the same draws."""
    stack = []
    skip = {"step", "init_game", "step_back"}
    orig_step, orig_init = game.step, game.init_game

    def _snapshot():
        return copy.deepcopy({k: v for k, v in game.__dict__.items() if k not in skip})

    def step(action):
        stack.append(_snapshot())
        return orig_step(action)

    def init_game():
        stack.clear()
        return orig_init()

    def step_back():
        if not stack:
            return False
        game.__dict__.update(stack.pop())
        return True

    game.step, game.init_game, game.step_back = step, init_game, step_back
    return game