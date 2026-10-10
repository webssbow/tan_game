"""
tan_rlcard.py -- RLCard environment for Tan.

Layers:
  TanGame   (your rules; untouched)
  TanRLGame (adapter: RLCard game protocol + turn arbitration)
  TanEnv    (RLCard Env: state encoding, action ids, payoffs)
"""
from __future__ import annotations

import random
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from rlcard.envs import Env
from rlcard.envs.registration import register

from tan_game import (
    ALL_RANKS, ALL_SUITS, Card, Action, Attack, Defend, AttackPass, DefendPass,
    Throwin, Continue, SwapTrump, TanGame,
)


@dataclass(frozen=True)
class DeclineSwap(Action):
    """Env-only action: the trump-2 holder is asked first and may refuse to swap."""
    pass


# --------------------------------------------------------------------------- #
# Fixed action-id space (independent of game state)
# --------------------------------------------------------------------------- #
ALL_CARDS: list[Card] = [Card(s, r) for r in ALL_RANKS for s in ALL_SUITS]      # 52
CARD_ID = {c: i for i, c in enumerate(ALL_CARDS)}

ATTACK_BASE     = 0                          # [0, 52)      Attack(card)
THROWIN_BASE    = 52                         # [52, 104)    Throwin(card)
DEFEND_BASE     = 104                        # [104, 2808)  Defend(attacking_card, card)
ATTACK_PASS_ID  = DEFEND_BASE + 52 * 52      # 2808
DEFEND_PASS_ID  = ATTACK_PASS_ID + 1         # 2809
CONTINUE_ID     = ATTACK_PASS_ID + 2         # 2810
SWAP_TRUMP_ID   = ATTACK_PASS_ID + 3         # 2811
DECLINE_SWAP_ID = ATTACK_PASS_ID + 4         # 2812
NUM_ACTIONS     = ATTACK_PASS_ID + 5         # 2813


def action_to_id(a: Action) -> int:
    match a:
        case Attack(_, card):       return ATTACK_BASE + CARD_ID[card]
        case Throwin(_, card):      return THROWIN_BASE + CARD_ID[card]
        case Defend(_, atk, card):  return DEFEND_BASE + 52 * CARD_ID[atk] + CARD_ID[card]
        case AttackPass():          return ATTACK_PASS_ID
        case DefendPass():          return DEFEND_PASS_ID
        case Continue():            return CONTINUE_ID
        case SwapTrump():           return SWAP_TRUMP_ID
        case DeclineSwap():         return DECLINE_SWAP_ID
    raise ValueError(f"cannot encode {a}")


def id_to_action(aid: int, player: int) -> Action:
    if aid < THROWIN_BASE:
        return Attack(player, ALL_CARDS[aid - ATTACK_BASE])
    if aid < DEFEND_BASE:
        return Throwin(player, ALL_CARDS[aid - THROWIN_BASE])
    if aid < ATTACK_PASS_ID:
        k = aid - DEFEND_BASE
        return Defend(player, ALL_CARDS[k // 52], ALL_CARDS[k % 52])
    return {
        ATTACK_PASS_ID:  AttackPass(player),
        DEFEND_PASS_ID:  DefendPass(player),
        CONTINUE_ID:     Continue(player),
        SWAP_TRUMP_ID:   SwapTrump(player),
        DECLINE_SWAP_ID: DeclineSwap(player),
    }[aid]


def _draw_seed(np_random) -> int:
    # RLCard hands us a np.random.RandomState (older) or Generator (newer)
    if hasattr(np_random, "integers"):
        return int(np_random.integers(0, 2**31 - 1))
    return int(np_random.randint(0, 2**31 - 1))


# --------------------------------------------------------------------------- #
# Game adapter: turn arbitration
# --------------------------------------------------------------------------- #
class TanRLGame:
    """
    Adapts TanGame (where several players may be able to act at once) to RLCard's
    strictly sequential protocol by deciding, after every action, exactly one
    acting player and their legal actions.

    Priority:
      1. SwapTrump holder (must Swap or DeclineSwap) -- absolute precedence.
      2. Defender, if they have anything to answer (when defender_first=True).
      3. Attackers that have NOT passed, ordered by `attacker_order`:
           'main_first'  : clockwise from the main attacker (Durak convention)
           'round_robin' : clockwise from the seat after the last actor
           'random'      : uniformly random among eligible attackers
      4. Defender (only reachable when defender_first=False).
    """

    def __init__(self, num_players=3, hand_size=6, defender_first=True,
                 attacker_order="main_first"):
        assert attacker_order in ("main_first", "round_robin", "random")
        self.num_players = num_players
        self.hand_size = hand_size
        self.defender_first = defender_first
        self.attacker_order = attacker_order
        self.np_random = np.random.RandomState()  # replaced by Env.seed()
        self.allow_step_back = False              # set by Env.__init__
        self.game: TanGame | None = None
        self.history: list = []
        self.current_player: int = 0
        self.legal: list[Action] = []
        self.last_actor: int = 0
        self.swap_declined = False

    # ---- RLCard game protocol --------------------------------------------
    def get_num_players(self): return self.num_players
    def get_num_actions(self): return NUM_ACTIONS
    def is_over(self):         return self.game.game_over
    def get_player_id(self):   return self.current_player

    def init_game(self):
        self.game = TanGame(self.num_players, self.hand_size,
                            random.Random(_draw_seed(self.np_random)))
        self.swap_declined = False
        self.last_actor = self.game.current_attacker
        self.history = []
        self._arbitrate()
        return self.get_state(self.current_player), self.current_player

    def step(self, action: Action):
        if self.allow_step_back:
            self.history.append((deepcopy(self.game), self.swap_declined, self.last_actor,
                                 self.current_player, list(self.legal)))
        if action not in self.legal:
            raise ValueError(f"illegal action {action}; legal: {self.legal}")
        if isinstance(action, DeclineSwap):
            self.swap_declined = True            # env-only, the game never sees it
        else:
            self.game.process_action(action)
        self.last_actor = action.player
        self._arbitrate()
        return self.get_state(self.current_player), self.current_player

    def step_back(self):
        if not self.history:
            return False
        (self.game, self.swap_declined, self.last_actor,
         self.current_player, self.legal) = self.history.pop()
        return True

    def get_state(self, player_id: int) -> dict:
        g = self.game
        legal = list(self.legal) if player_id == self.current_player else []
        return {
            "player": player_id,
            "hand": list(g.hands[player_id]),
            "hand_sizes": [len(h) for h in g.hands],
            "attack_pile": list(g.attack_pile),
            "discard_pile": list(g.discard_pile),
            "trump": g.trump,
            "draw_pile_size": len(g.draw_pile),
            "current_attacker": g.current_attacker,
            "defender": g.defender,
            "passed": list(g.passed),
            "active": list(g.current_players),
            "free_attack_pile": g.free_attack_pile,
            "can_throwin": g.can_throwin,
            "swap_offer": any(isinstance(a, SwapTrump) for a in legal),
            "legal_actions": legal,
        }

    # ---- arbitration -----------------------------------------------------
    def _arbitrate(self):
        g = self.game
        if g.game_over:
            self.current_player, self.legal = self.last_actor, []
            return

        valid = g.valid_actions()

        # 1. SwapTrump first. Holder must explicitly swap or decline.
        swap = next((a for a in valid if isinstance(a, SwapTrump)), None)
        if swap is not None and not self.swap_declined:
            self.current_player = swap.player
            self.legal = [swap, DeclineSwap(swap.player)]
            return

        by_player: dict[int, list[Action]] = {}
        for a in valid:
            if not isinstance(a, SwapTrump):
                by_player.setdefault(a.player, []).append(a)

        # 2. Defender answers before anything else gets thrown in.
        if self.defender_first and g.defender in by_player:
            self.current_player, self.legal = g.defender, by_player[g.defender]
            return

        # 3. Attackers who have not passed yet.
        candidates = [p for p in by_player if p != g.defender and not g.passed[p]]
        if candidates:
            self.current_player = self._pick_attacker(candidates)
            self.legal = by_player[self.current_player]
            return

        # 4. Defender last (attackers-first convention).
        if g.defender in by_player:
            self.current_player, self.legal = g.defender, by_player[g.defender]
            return

        raise RuntimeError(f"nobody can act but game is not over:\n{g}")

    def _pick_attacker(self, candidates: list[int]) -> int:
        n = self.num_players
        if self.attacker_order == "random":
            return int(candidates[self.np_random.randint(len(candidates))])
        start = (self.game.current_attacker if self.attacker_order == "main_first"
                 else (self.last_actor + 1) % n)
        for k in range(n):
            p = (start + k) % n
            if p in candidates:
                return p
        raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# RLCard Env
# --------------------------------------------------------------------------- #
DEFAULT_GAME_CONFIG = {
    "game_num_players": 3,
    "game_hand_size": 6,
    "game_defender_first": True,
    "game_attacker_order": "main_first",
}


def obs_dim(num_players: int) -> int:
    # 6 card planes + trump suit one-hot + 5 features per seat + 4 misc flags
    return 6 * 52 + 4 + 5 * num_players + 4


class TanEnv(Env):
    def __init__(self, config):
        self.name = "tan"
        self.default_game_config = DEFAULT_GAME_CONFIG
        cfg = {**DEFAULT_GAME_CONFIG,
               **{k: v for k, v in config.items() if k in DEFAULT_GAME_CONFIG}}
        self.game = TanRLGame(
            num_players=cfg["game_num_players"],
            hand_size=cfg["game_hand_size"],
            defender_first=cfg["game_defender_first"],
            attacker_order=cfg["game_attacker_order"],
        )
        super().__init__(config)   # sets allow_step_back, np_random, num_players, num_actions
        self.state_shape = [[obs_dim(self.num_players)] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    # ---- observation -----------------------------------------------------
    def _encode_obs(self, s: dict) -> np.ndarray:
        me, n = s["player"], self.num_players
        planes = np.zeros((6, 52), dtype=np.float32)
        for c in s["hand"]:
            planes[0, CARD_ID[c]] = 1                 # my hand
        for atk, dfn in s["attack_pile"]:
            if dfn is None:
                planes[1, CARD_ID[atk]] = 1           # uncovered attack cards
            else:
                planes[2, CARD_ID[atk]] = 1           # covered attack cards
                planes[3, CARD_ID[dfn]] = 1           # defending cards
        for c in s["discard_pile"]:
            planes[4, CARD_ID[c]] = 1                 # out of play (perfect memory)
        planes[5, CARD_ID[s["trump"]]] = 1            # face-up trump card

        suit = np.zeros(4, dtype=np.float32)
        suit[ALL_SUITS.index(s["trump"].suit)] = 1

        seats = np.zeros((n, 5), dtype=np.float32)    # relative seating: row 0 = me
        for k in range(n):
            p = (me + k) % n
            seats[k] = [s["hand_sizes"][p] / self.game.hand_size,
                        p == s["defender"], p == s["current_attacker"],
                        s["passed"][p], s["active"][p]]

        misc = np.array([s["draw_pile_size"] / 52, s["free_attack_pile"],
                         s["can_throwin"], s["swap_offer"]], dtype=np.float32)
        return np.concatenate([planes.ravel(), suit, seats.ravel(), misc])

    def _extract_state(self, state: dict) -> dict:
        return {
            "obs": self._encode_obs(state),
            "legal_actions": OrderedDict((action_to_id(a), None) for a in state["legal_actions"]),
            "raw_obs": state,
            "raw_legal_actions": list(state["legal_actions"]),
            "action_record": self.action_recorder,
        }

    def _get_legal_actions(self):
        return OrderedDict((action_to_id(a), None) for a in self.game.legal)

    def _decode_action(self, action_id) -> Action:
        return id_to_action(int(action_id), self.game.current_player)

    # ---- rewards ---------------------------------------------------------
    def get_payoffs(self):
        """Rank-based, zero-sum: first out = +1, last one holding cards = -1."""
        g, n = self.game.game, self.num_players
        order = list(g.finished_players) + [p for p in range(n) if g.current_players[p]]
        payoffs = np.zeros(n, dtype=np.float32)
        for rank, p in enumerate(order):
            payoffs[p] = 1.0 - 2.0 * rank / (n - 1)
        return payoffs

    def get_perfect_information(self):
        g = self.game.game
        return {
            "hands": [list(h) for h in g.hands],
            "draw_pile": list(g.draw_pile),
            "trump": g.trump,
            "attack_pile": list(g.attack_pile),
            "discard_pile": list(g.discard_pile),
            "current_player": self.game.current_player,
            "legal_actions": list(self.game.legal),
        }


# Register so rlcard.make('tan', ...) works once this module has been imported.
try:
    register(env_id="tan", entry_point="tan_rlcard:TanEnv")
except ValueError:
    pass  # already registered (e.g. module re-imported in a notebook)


def make_tan_env(config: dict | None = None) -> TanEnv:
    """Bypass the registry if you'd rather not rely on import paths."""
    cfg = {"allow_step_back": False, "seed": None}
    cfg.update(config or {})
    return TanEnv(cfg)


if __name__ == "__main__":
    # smoke test: random agents, check card conservation & payoffs
    import rlcard
    from rlcard.agents import RandomAgent

    env = rlcard.make("tan", config={"seed": 0, "game_num_players": 3})
    env.set_agents([RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)])
    for i in range(200):
        trajectories, payoffs = env.run(is_training=False)
        env.game.game.assert_52_cards()
    print("obs dim:", env.state_shape[0], "| actions:", env.num_actions)
    print("last payoffs:", payoffs, "| steps in last game:", len(env.action_recorder))