import random
from dataclasses import dataclass
from typing import Union
from typing import NamedTuple

SUIT_SYMBOLS = {"D": "♦", "C": "♣", "H": "♥", "S": "♠"}

class Card(NamedTuple):
    suit: str
    rank: str

    def __str__(self):
        return f"{self.rank}{SUIT_SYMBOLS[self.suit]}"

    def __repr__(self):
        return self.__str__()

ALL_RANKS = list('23456789TJQKA')
ALL_SUITS = list('DCHS')
VALID_TRUMP_RANKS = set('3456789')
RANK_VALUE = {r: i for i, r in enumerate(ALL_RANKS)}

def valid_attack(c1: Card, c2: Card, trump_suit: str): # whether c2 can attack c1
    comp = c2.suit == c1.suit and RANK_VALUE[c2.rank] > RANK_VALUE[c1.rank] # two cards (of the same suit), c2 can beat c1
    if c1.suit == trump_suit:
        return comp # they have to both be trump
    else:
        return c2.suit == trump_suit or comp # c2 can be trump, or regular comp

@dataclass(frozen=True)
class Action: player: int

@dataclass(frozen=True)
class Attack(Action): card: Card # attack next player with a card

@dataclass(frozen=True)
class Defend(Action): # use card to defend against attacking_card
    attacking_card: Card 
    card: Card

@dataclass(frozen=True)
class AttackPass(Action): pass # a signal that other attacking players do not want to play any moves

@dataclass(frozen=True)
class DefendPass(Action): pass # a signal that the defending player will pick up the cards

@dataclass(frozen=True)
class Throwin(Action): card: Card # during the discard phase, the player can throw in cards in their own hand that have the same rank as any card in the to-be-discarded pile

@dataclass(frozen=True)
class Continue(Action): pass # if the defender wins, and has finished throwin card phase, they must declare Continue to continue the round.

@dataclass(frozen=True)
class SwapTrump(Action): pass # action only allowed at the start of the round. if someone has a 2 of the trump suit, they can swap it with the trump card.

class TanGame:
    def __str__(self):
        return f"{self.__class__.__name__}: {vars(self)}"

    def __repr__(self):
        return self.__str__()

    def __init__(self, num_players: int, hand_size: int, rng: random.Random):
        assert num_players >= 2
        self.num_players = num_players
        self.hand_size = hand_size
        self.rng = rng

        self.start_game()

    def start_game(self):
        assert self.num_players * self.hand_size < 52, "not enough cards to deal hands and draw a trump"
        self.hands: list[list[Card]] = [[] for _ in range(self.num_players)]
        self.draw_pile: list[Card] = [Card(s, r) for r in ALL_RANKS for s in ALL_SUITS]
        self.discard_pile: list[Card] = []
        self.round_started = False
        self.attack_pile: list[tuple[Card, Card | None]] = []
        self.passed = [False for _ in range(self.num_players)]
        self.can_throwin = False
        self.current_players = [True for _ in range(self.num_players)]
        self.finished_players: list[int] = [] # seat indices, in the order they finished
        self.free_attack_pile = False
        self.current_attacker = 0
        self.defender = 1

        while True:
            self.rng.shuffle(self.draw_pile)
            for _ in range(self.hand_size):
                for p in range(self.num_players):
                    self.hands[p].append(self.draw_pile.pop())
            if any(c.rank in VALID_TRUMP_RANKS for c in self.draw_pile):
                break
            # no valid trump left in the draw pile: collect the hands and redeal
            for hand in self.hands:
                self.draw_pile.extend(hand)
                hand.clear()

        while True:
            self.trump = self.draw_pile[-1]
            if self.trump.rank in VALID_TRUMP_RANKS:
                self.draw_pile.pop()
                self.draw_pile.insert(0, self.trump)
                break
            self.rng.shuffle(self.draw_pile)

    def next_active(self, seat: int):   # works even if seat has finished
        N = len(self.hands)
        for k in range(1, N + 1):
            s = (seat + k) % N
            if self.current_players[s]:
                return s
        else: raise ValueError("could not find next seat")

    @property
    def game_over(self):
        return sum(self.current_players) <= 1

    def attackers(self):
        return [p for p in range(len(self.hands)) if self.current_players[p] and p != self.defender]

    def finish(self, p):
        self.current_players[p] = False
        self.passed[p] = True      # a finished seat can never block "everyone passed"
        self.finished_players.append(p)

    def assert_52_cards(self):
        cards_in_hand = sum(len(hand) for hand in self.hands)
        cards_in_attack_pile = sum(1 for pair in self.attack_pile for c in pair if c is not None)
        total = cards_in_hand + len(self.draw_pile) + len(self.discard_pile) + cards_in_attack_pile
        assert total == 52, f"number of cards counted = {total} != 52. total cards in hand: {cards_in_hand} | draw pile: {len(self.draw_pile)} | discard pile: {len(self.discard_pile)} | attack pile: {cards_in_attack_pile}"

    def valid_actions(self):
        if self.game_over: return []

        valid = []
        # swap trump
        if not self.round_started:
            for i, hand in enumerate(self.hands):
                if Card(self.trump.suit, '2') in hand:
                    valid.append(SwapTrump(i))
                    break

        if not self.attack_pile: # start of attack
            for card in self.hands[self.current_attacker]:
                valid.append(Attack(self.current_attacker, card))
        elif self.free_attack_pile: # defending player already won, every attacker can just dump valid cards if they want
            flattened_attack_pile_ranks = {c.rank for pair in self.attack_pile for c in pair if c is not None}
            for i, hand in enumerate(self.hands):
                if not self.passed[i]: valid.append(AttackPass(i))
                for card in hand:
                    if card.rank in flattened_attack_pile_ranks:
                        valid.append(Attack(i, card))
        elif self.can_throwin: # every attacker has passed and the defender has defended against all attacking cards
            flattened_attack_pile_ranks = {c.rank for pair in self.attack_pile for c in pair if c is not None} # none of them should be None, but its here for the type checker
            for card in self.hands[self.defender]:
                if card.rank in flattened_attack_pile_ranks:
                    valid.append(Throwin(self.defender, card))
            valid.append(Continue(self.defender))
        else: # regular attacking phase
            for i, hand in enumerate(self.hands):
                if i == self.defender:
                    if self.passed[i]: continue # defender is not allowed to do anything if they pass
                    else: # defender has not passed yet
                        if not all(c[1] for c in self.attack_pile): # if the defender has not defended against all cards yet, they are allowed to pass. but if they have defended against all cards they are not allowed to
                            valid.append(DefendPass(i))
                        for (attacking_card, defending_card) in self.attack_pile:
                            if defending_card: continue
                            for card in hand:
                                if valid_attack(attacking_card, card, self.trump.suit):
                                    valid.append(Defend(i, attacking_card, card))
                else:
                    if not self.passed[i]: valid.append(AttackPass(i))
                    flattened_attack_pile_ranks = {c.rank for pair in self.attack_pile for c in pair if c is not None}
                    for card in hand:
                        if card.rank in flattened_attack_pile_ranks:
                            valid.append(Attack(i, card))
        return valid

    def refill(self, player: int):
        for _ in range(self.hand_size):
            if not self.draw_pile: break
            if len(self.hands[player]) >= self.hand_size: break
            self.hands[player].append(self.draw_pile.pop())

    def refill_or_declare_win(self, player: int):
        assert len(self.hands[player]) == 0, f"on refill, expected player {player} hand to be empty, found non-empty hand {self.hands[player]}"
        if self.draw_pile:
            self.refill(player)
            return
        # no more cards in draw pile, hence declare win
        self.finish(player)
        if player == self.defender:
            # defender wins; attackers can now dump matching cards, even if they had passed before
            self.free_attack_pile = True
            for p in self.attackers():
                self.passed[p] = False

    def advance_round(self, defender_won: bool):
        cards = [c for pair in self.attack_pile for c in pair if c is not None]
        if defender_won:
            self.discard_pile.extend(cards)
        else:
            self.hands[self.defender].extend(cards)
        self.attack_pile = []

        # refill: attacker, then defender, then clockwise from the defender
        n = len(self.hands)
        order = [self.current_attacker, self.defender]
        order += [(self.defender + k) % n for k in range(1, n) if (self.defender + k) % n not in order]
        for p in order:
            if self.current_players[p]:
                self.refill(p)

        # anyone still empty-handed after refilling (draw pile is empty) has won
        for p in range(n):
            if self.current_players[p] and not self.hands[p]:
                self.finish(p)

        self.passed = [not a for a in self.current_players]
        self.can_throwin = False
        self.free_attack_pile = False
        if self.game_over:
            return

        if defender_won and self.current_players[self.defender]:
            nxt = self.defender                       # defender won -> they attack next
        else:
            nxt = self.next_active(self.defender)     # defender picked up (or finished) -> skip them
        self.current_attacker, self.defender = nxt, self.next_active(nxt)

    # it is assumed that the action is returned from valid_actions and is a valid action. i still add some braindead checks tho for debugging purposes
    def process_action(self, action: Action):
        match action:
            case Attack(player, card):
                self.passed[player] = False
                self.hands[player].remove(card)
                self.attack_pile.append((card, None))
                if not self.hands[player]:
                    self.refill_or_declare_win(player)
            case Defend(player, attacking_card, card):
                assert player == self.defender, f"someone ({player}) who wasnt the defender ({self.defender}) used a Defend action"
                self.hands[self.defender].remove(card)
                flattened_attack_pile_ranks = {c.rank for pair in self.attack_pile for c in pair if c is not None}
                if card.rank not in flattened_attack_pile_ranks:
                    self.passed = [not a for a in self.current_players]
                for i, (c, _) in enumerate(self.attack_pile):
                    if c == attacking_card:
                        self.attack_pile[i] = (c, card)
                        break
                else:
                    raise ValueError(f"{attacking_card} not found in attack pile ({self.attack_pile})")
                if not self.hands[self.defender]:
                    self.refill_or_declare_win(self.defender)
            case AttackPass(player):
                assert not self.passed[player], f"attacker {player} tried to pass when they already passed before"
                self.passed[player] = True
            case DefendPass(player):
                assert player == self.defender, f"someone ({player}) who wasnt the defender ({self.defender}) used a DefendPass action"
                assert not self.passed[self.defender], f"defender {player} tried to pass when they already passed before"
                self.passed[self.defender] = True
                for p in self.attackers():      # attackers may now pile on new matching cards
                    self.passed[p] = False
            case Throwin(player, card):
                assert player == self.defender, f"someone ({player}) who wasnt the defender ({self.defender}) used a Throwin action"
                assert self.can_throwin, f"defender ({self.defender}) used Throwin action when can_throwin == False"
                self.hands[self.defender].remove(card)
                self.attack_pile.append((card, None))
            case Continue(player):
                assert player == self.defender, f"someone ({player}) who wasnt the defender ({self.defender}) used a Continue action"
                assert self.can_throwin, f"defender ({self.defender}) used Continue action when can_throwin == False"
                self.advance_round(True)
            case SwapTrump(player):
                assert not self.round_started, f"player {player} illegally swapped a trump after the start of the round"
                trump2 = Card(self.trump.suit, "2")
                self.hands[player].remove(trump2)
                self.hands[player].append(self.trump)
                self.trump = trump2
                self.draw_pile[0] = trump2
                self.current_attacker = player
                self.defender = self.next_active(player)
            case a:
                raise ValueError(f"unknown action {a}")
        
        if self.game_over or not self.attack_pile: return
        atk_done = all(self.passed[p] for p in self.attackers())
        if self.free_attack_pile:
            if atk_done: self.advance_round(True)
        elif self.passed[self.defender]:
            if atk_done: self.advance_round(False)
        elif atk_done and all(d is not None for _, d in self.attack_pile):
            self.can_throwin = True
        self.round_started = True # any action played means that the round has started