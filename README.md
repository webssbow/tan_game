This repository contains an environment, trained bot, and GUI to play the Vietnamese game Tấn.

[**Download the latest release here**](https://github.com/webssbow/tan_game/releases/latest)

# Build

```
pip install rlcard[torch]
python tan_gui.py
```

# Rules of Tấn

*(note: **rank** refers to the number on the card)*

## Origin
Tấn is the Vietnamese adaptation of the Russian card game **Durak**. It spread to Vietnam and became one of the most popular Western-deck games there. As in Durak, the loser is the one left holding cards when everyone else has emptied their hand.

## Overview
- **Players:** 2-4 (env default: 3).
- **Deck:** standard 52-card deck.
- **Hand size:** refill target is 8 (for simplicity, in env default is 6).
- **Direction:** clockwise.
- **Goal:** be the **first to empty your hand**. The last player holding cards is the "fool."

## Card Ranking
- **Ranks:** 2 < 3 < 4 < ... < 10 < J < Q < K < A (Ace high).
- **Suits:** no intrinsic order; the trump suit outranks all others.

## Setup
1. Deal `hand_size` (default: 6) cards to each player.
2. Place the rest of the deck face down as the **draw pile**.
3. Flip one card from the draw pile face-up and tuck it under the pile: this is the **trump card**, and its suit is the **trump suit**.
   - The trump card's **rank must be 3-9**. If it is 2, 10, J, Q, K, or A, reshuffle and redraw until valid.
4. **Trump-2 swap (opening only):** whoever holds the 2 of the trump suit may swap it for the face-up trump card. If they do, that player opens the first attack.

## Trump (a.k.a. "Tấn") Cards
- Any trump card beats any non-trump card. For example, if the trump suit is ♠, 3♠ > K♦
- Among trump cards, normal rank order applies.

## Turn Structure
Each **round** has one **defender** and one or more **attackers**. Play proceeds:

1. **Attack phase**: attackers take turns clockwise, each either playing an **attack card** or **passing**, until all attackers have passed.
   - The **main attacker** plays any card first.
   - Subsequent **throw-ins** must match the rank of some card already on the pile (attack or defense).
2. **Defense phase**: the defender plays **one** card to beat the current uncovered attack card, or gives up.
3. Repeat 1-2 until the round ends.

*(note: this is slightly different than regular Tấn rules, but is made this way so that the play ordering is more rigorous, which is needed for the bot logic)*

## Attacking & Defending
**Attack**
- The first attacker may start attacking with any card.
- Throw-ins (by any attacker) must match a rank already present on the pile.

**Defense**: a defender's card must
- be the **same suit** and **higher rank** than the card it covers, **or**
- be a **trump** (which beats any non-trump).

**Round outcome**
- **Defender beats everything →** all pile cards are discarded; the defender becomes the **next main attacker**.
- **Defender gives up →** the defender takes the entire pile into their hand; the player to the defender's **right** becomes the next main attacker.

## Drawing / Refilling
- Players below `hand_size` refill from the draw pile **immediately**, even mid-round (e.g., after defending).
- Refill order: **main attacker → defender → clockwise from the player after the defender**.
- If any attacker or defender runs out of cards *while the draw pile is non-empty*, they must immediately refill `hand_size` cards, regardless of whether the attacking phase has ended yet.
- If the draw pile empties, play continues with current hands only.

## End of Game
- A player **empties their hand → wins immediately**. (If the defender empties their hand while defending with the draw pile exhausted, they win automatically regardless of how many attack cards remain.)
- The last player still holding cards is the loser.