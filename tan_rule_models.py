"""
tan_rule_models.py -- Rule-based baseline agents for Tấn (RLCard).

State fields used (from TanRLGame.get_state):
  hand            list[Card]                 my cards
  hand_sizes      list[int]                  per seat
  attack_pile     list[(Card, Card|None)]    (attacking card, covering card or None)
  discard_pile    list[Card]                 permanently out of play
  trump           Card                       face-up trump card (suit = trump suit)
  draw_pile_size  int
  current_attacker, defender                 seat ids
  passed, active  list[bool]
  free_attack_pile bool                      True -> I'm opening a fresh attack
  can_throwin     bool
  legal_actions   list[Action]               (same objects as raw_legal_actions)
"""
from __future__ import annotations

from collections import Counter

import rlcard
from rlcard.models.model import Model

from tan_game import (
    ALL_RANKS, RANK_VALUE, Card,
    Attack, Defend, AttackPass, DefendPass, Throwin, Continue, SwapTrump,
)
import tan_rlcard  # noqa: F401 -- side effect: registers env id 'tan'
from tan_rlcard import DeclineSwap  # noqa: F401 (kept for completeness / type checks)

HIGH_RANK = RANK_VALUE["Q"]   # Q, K, A are "high"
MID_RANK = RANK_VALUE["T"]    # <= T is "cheap to give away"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def is_trump(card: Card, trump_suit: str) -> bool:
    return card.suit == trump_suit


def power(card: Card, trump_suit: str) -> int:
    """Total ordering of card strength: any trump beats any non-trump."""
    return RANK_VALUE[card.rank] + (13 if card.suit == trump_suit else 0)


def _first(legal, cls):
    return next((a for a in legal if isinstance(a, cls)), None)


def _all(legal, cls):
    return [a for a in legal if isinstance(a, cls)]


def _uncovered(obs) -> list[Card]:
    return [atk for atk, dfn in obs["attack_pile"] if dfn is None]


def _pile_cards(obs) -> list[Card]:
    out = []
    for atk, dfn in obs["attack_pile"]:
        out.append(atk)
        if dfn is not None:
            out.append(dfn)
    return out


# --------------------------------------------------------------------------- #
# V1: greedy baseline
# --------------------------------------------------------------------------- #
class TanRuleAgentV1:
    """
    Dumb-but-legal baseline:
      * always swap the trump 2,
      * always attack / add the lowest-power legal card, never pass voluntarily,
      * always defend with the lowest-power legal cover, pick up only when forced,
      * throw in everything it can, then Continue.
    """

    def __init__(self):
        self.use_raw = True

    def step(self, state):
        legal = state["raw_legal_actions"]
        obs = state["raw_obs"]
        ts = obs["trump"].suit

        if len(legal) == 1:
            return legal[0]

        swap = _first(legal, SwapTrump)
        if swap:
            return swap

        defends = _all(legal, Defend)
        if defends:
            return min(defends, key=lambda d: power(d.card, ts))
        if _first(legal, DefendPass):
            return _first(legal, DefendPass)

        throwins = _all(legal, Throwin)
        if throwins:
            return min(throwins, key=lambda t: power(t.card, ts))

        attacks = _all(legal, Attack)
        if attacks:
            return min(attacks, key=lambda a: power(a.card, ts))

        for cls in (Continue, AttackPass):
            a = _first(legal, cls)
            if a:
                return a
        return legal[0]

    def eval_step(self, state):
        return self.step(state), {}


# --------------------------------------------------------------------------- #
# V2: heuristic baseline encoding the strategy notes
# --------------------------------------------------------------------------- #
class TanRuleAgentV2:
    """
    Heuristic Tấn player.

    Attacking
      - Opening: lowest non-trump card, preferring ranks held in multiples
        (pairs let me keep piling on). Trumps only if nothing else.
      - Endgame (deck empty): if I hold a trump that nothing in play can beat
        (all higher trumps are discarded or in my hand), lead it -> forced pickup.
      - Adding to an attack: pile on cheap non-trumps (<= T) or pairs; never
        give away trumps or high cards while the deck is alive. If the defender
        has already burned a trump early, PASS so the trump is discarded.

    Defending
      - Check whether every uncovered card can actually be covered (greedy
        assignment, hardest attack first). If not, pick up immediately instead
        of leaking cards/information.
      - Cover with the cheapest card; slight preference for ranks that are
        already heavily "seen" (discard + table + my hand) so attackers have
        fewer continuation cards.
      - Early game, small pile: don't burn a trump on a cheap non-trump attack
        unless I'm trump-rich; and happily pick up a trump attack to bank it.

    Throw-ins
      - If the defender won (all cards covered -> discard): shed everything.
      - If the defender is picking up: shed non-trumps; trumps only when deck
        is empty (pure race to zero cards).
    """

    def __init__(self, early_deck_threshold: int | None = None,
                 trump_reserve: int = 2, max_bait_pile: int = 2):
        self.use_raw = True
        self.early_deck_threshold = early_deck_threshold  # None -> derived from #players
        self.trump_reserve = trump_reserve                # keep spending trumps if I hold > this
        self.max_bait_pile = max_bait_pile                # only voluntarily pick up small piles

    # ----------------------------------------------------------------- #
    def step(self, state):
        legal = state["raw_legal_actions"]
        obs = state["raw_obs"]
        if not legal:
            raise ValueError("no legal actions")
        if len(legal) == 1:
            return legal[0]

        swap = _first(legal, SwapTrump)
        if swap:                                   # 2♠ -> 8♠ and go first: always good
            return swap

        defends = _all(legal, Defend)
        if defends or _first(legal, DefendPass):
            action = self._defend(obs, legal, defends)
        elif _all(legal, Throwin):
            action = self._throwin(obs, legal)
        elif _all(legal, Attack) or _first(legal, AttackPass):
            action = self._attack(obs, legal)
        else:
            action = _first(legal, Continue) or legal[0]

        return action if action in legal else legal[0]   # safety net

    def eval_step(self, state):
        return self.step(state), {}

    # ----------------------------------------------------------------- #
    def _early(self, obs) -> bool:
        thr = self.early_deck_threshold
        if thr is None:
            thr = 2 * len(obs["hand_sizes"])      # roughly "more than one refill round left"
        return obs["draw_pile_size"] >= thr

    def _seen_copies(self, obs, rank: str) -> int:
        seen = list(obs["discard_pile"]) + _pile_cards(obs) + list(obs["hand"])
        return sum(1 for c in seen if c.rank == rank)

    def _unbeatable(self, card: Card, obs) -> bool:
        ts = obs["trump"].suit
        if not is_trump(card, ts):
            return False
        gone = set(obs["discard_pile"]) | set(obs["hand"])
        higher = (Card(ts, r) for r in ALL_RANKS if RANK_VALUE[r] > RANK_VALUE[card.rank])
        return all(h in gone for h in higher)

    # ---- defence ------------------------------------------------------
    def _assign(self, obs, defends):
        """Greedy cover assignment. Returns list[(atk, Defend)] or None if infeasible."""
        ts = obs["trump"].suit
        covers: dict[Card, list[Defend]] = {}
        for d in defends:
            covers.setdefault(d.attacking_card, []).append(d)

        uncovered = _uncovered(obs)
        if any(atk not in covers for atk in uncovered):
            return None

        def cover_score(d: Defend):
            # cheaper is better; ranks already heavily seen are slightly preferred
            return power(d.card, ts) - 0.5 * self._seen_copies(obs, d.card.rank)

        used: set[Card] = set()
        plan = []
        for atk in sorted(uncovered, key=lambda c: power(c, ts), reverse=True):
            opts = [d for d in covers[atk] if d.card not in used]
            if not opts:
                return None
            best = min(opts, key=cover_score)
            used.add(best.card)
            plan.append((atk, best))
        return plan

    def _defend(self, obs, legal, defends):
        dpass = _first(legal, DefendPass)
        if not defends:
            return dpass

        ts = obs["trump"].suit
        plan = self._assign(obs, defends)
        if plan is None:
            return dpass or min(defends, key=lambda d: power(d.card, ts))

        if dpass and self._early(obs) and len(_pile_cards(obs)) <= self.max_bait_pile:
            n_trumps = sum(is_trump(c, ts) for c in obs["hand"])
            for atk, d in plan:
                # Bank an early trump attack instead of over-trumping it.
                if is_trump(atk, ts):
                    return dpass
                # Don't burn a trump on a cheap non-trump unless trump-rich.
                if (is_trump(d.card, ts) and RANK_VALUE[atk.rank] < HIGH_RANK
                        and n_trumps <= self.trump_reserve):
                    return dpass

        # Cover the hardest attacking card first (plan is sorted that way).
        return plan[0][1]

    # ---- attack -------------------------------------------------------
    def _attack(self, obs, legal):
        apass = _first(legal, AttackPass)
        attacks = _all(legal, Attack)
        if not attacks:
            return apass

        ts = obs["trump"].suit
        hand = obs["hand"]
        endgame = obs["draw_pile_size"] == 0
        counts = Counter(c.rank for c in hand)

        if obs["free_attack_pile"]:
            if endgame:
                forcing = [a for a in attacks if self._unbeatable(a.card, obs)]
                if forcing:
                    return min(forcing, key=lambda a: power(a.card, ts))

            def open_score(a: Attack):
                c = a.card
                s = power(c, ts)
                if not is_trump(c, ts):
                    s -= 1.5 * (counts[c.rank] - 1)   # pairs/triples: keep pressure
                return s
            return min(attacks, key=open_score)

        # ---- piling on to an existing attack
        defender_used_trump = any(
            dfn is not None and is_trump(dfn, ts) for _, dfn in obs["attack_pile"])
        if defender_used_trump and self._early(obs) and apass:
            return apass                       # let them lose that trump to the discard

        defender_hand = obs["hand_sizes"][obs["defender"]]

        def worth_adding(a: Attack) -> bool:
            c = a.card
            if is_trump(c, ts):
                # Only dump trumps when racing to empty the hand.
                return endgame and (len(hand) <= 2 or defender_hand <= 2)
            return endgame or RANK_VALUE[c.rank] <= MID_RANK or counts[c.rank] >= 2

        cands = [a for a in attacks if worth_adding(a)]
        if not cands:
            return apass or min(attacks, key=lambda a: power(a.card, ts))
        return min(cands, key=lambda a: power(a.card, ts))

    # ---- throw-in -----------------------------------------------------
    def _throwin(self, obs, legal):
        ts = obs["trump"].suit
        throwins = _all(legal, Throwin)
        endgame = obs["draw_pile_size"] == 0
        pickup = bool(_uncovered(obs))        # defender gave up -> cards go to them

        if pickup:
            ok = [t for t in throwins if endgame or not is_trump(t.card, ts)]
        else:
            ok = list(throwins)               # going to the discard: shed freely

        if ok:
            return min(ok, key=lambda t: power(t.card, ts))

        # nothing worth throwing: end the phase with whatever the env offers
        for cls in (Continue, AttackPass, DefendPass):
            a = _first(legal, cls)
            if a:
                return a
        return next(a for a in legal if not isinstance(a, Throwin))


# --------------------------------------------------------------------------- #
# RLCard Model wrappers
# --------------------------------------------------------------------------- #
class TanRuleModelV1(Model):
    def __init__(self, env_config: dict | None = None):
        env = rlcard.make("tan", config=env_config or {})
        agent = TanRuleAgentV1()
        self.rule_agents = [agent for _ in range(env.num_players)]

    @property
    def agents(self):
        return self.rule_agents


class TanRuleModelV2(Model):
    def __init__(self, env_config: dict | None = None):
        env = rlcard.make("tan", config=env_config or {})
        agent = TanRuleAgentV2()
        self.rule_agents = [agent for _ in range(env.num_players)]

    @property
    def agents(self):
        return self.rule_agents


# Optional: make rlcard.models.load('tan-rule-v2') work.
try:
    from rlcard.models.registration import register as _register_model
    _register_model(model_id="tan-rule-v1", entry_point="tan_rule_models:TanRuleModelV1")
    _register_model(model_id="tan-rule-v2", entry_point="tan_rule_models:TanRuleModelV2")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Smoke test / quick benchmark
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from rlcard.agents import RandomAgent
    from rlcard.utils import tournament

    env = rlcard.make("tan", config={"seed": 0, "game_num_players": 3})

    # V2 in seat 0 versus random opponents
    env.set_agents([TanRuleAgentV2()] + [RandomAgent(num_actions=env.num_actions)
                                         for _ in range(env.num_players - 1)])
    print("V2 vs random :", tournament(env, 500))

    # V2 in seat 0 versus V1 greedy
    env.set_agents([TanRuleAgentV2()] + [TanRuleAgentV1() for _ in range(env.num_players - 1)])
    print("V2 vs V1     :", tournament(env, 500))

    # legality check over many games
    for _ in range(1000):
        env.run(is_training=False)
        env.game.game.assert_52_cards()
    print("ok")