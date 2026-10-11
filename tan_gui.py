"""
tan_gui.py -- clickable Tkinter UI for playing Tan (your RLCard env) against
RLCard-compatible agents (RandomAgent, DQNAgent, ...).

Usage:
    python tan_gui.py
"""
from __future__ import annotations
import sys

# rlcard calls `sys.executable -m pip freeze` at import time to enumerate
# installed packages. In a frozen PyInstaller binary sys.executable IS this
# binary, so every launch spawns another copy of itself -- an infinite loop.
# Catch that re-spawn here, before any other imports, and exit immediately.
if getattr(sys, 'frozen', False) and len(sys.argv) > 1 and sys.argv[1] == '-m':
    sys.exit(0)

import multiprocessing
multiprocessing.freeze_support()  # required for PyInstaller + PyTorch on macOS

import tkinter as tk
from tkinter import messagebox

from tan_game import (
    RANK_VALUE, SUIT_SYMBOLS, Attack, Defend, AttackPass, DefendPass, Throwin,
    Continue, SwapTrump,
)
from tan_rlcard import DeclineSwap, action_to_id, id_to_action

from tan_common import load_torch_agent
import os, sys


def get_resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

def create_dmc_agent(seat=0, savedir="experiments/tan_dmc", xpid="tan"):
    run_dir = os.path.join(savedir, xpid)
    best_model_name = f"{seat}_2081369600.pth"
    
    # Try multiple locations (for dev vs bundled app)
    ckpt_path = None
    
    # 1. Try as bundled resource (inside the app)
    bundled_path = get_resource_path(os.path.join("experiments", "tan_dmc", "tan", best_model_name))
    if os.path.exists(bundled_path):
        ckpt_path = bundled_path
    else:
        # 2. Try relative to current working directory (development)
        dev_path = os.path.join(run_dir, best_model_name)
        if os.path.exists(dev_path):
            ckpt_path = dev_path
    
    if ckpt_path is None:
        print(f"Error: Could not find model {best_model_name}")
        print(f"  Tried: {bundled_path}")
        print(f"  Tried: {os.path.join(run_dir, best_model_name)}")
        # In a GUI app, you might want to show a message box instead of exit
        import tkinter.messagebox as msgbox
        msgbox.showerror("Model Not Found", 
                        f"Could not load model: {best_model_name}\n"
                        f"Please ensure the model file is included in the app bundle.")
        sys.exit(1)
    
    print(f"Loading model from: {ckpt_path}")
    dmc_agent = load_torch_agent(ckpt_path, "cpu")
    return dmc_agent


RED_SUITS = {"H", "D"}

CARD_W, CARD_H = 70, 100          # hand / table cards
MINI_W, MINI_H = 34, 48           # opponent / discard face-down stack cards
TRUMP_FILL = "#b1eea7"            # mild green highlight
TABLE_BG = "#1b6b34"              # felt green background

CANVAS_W, CANVAS_H = 960, 580


def rounded_rect(canvas, x1, y1, x2, y2, r=10, **kwargs):
    points = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def draw_card(canvas, cx, cy, card=None, face_down=False, w=CARD_W, h=CARD_H,
              trump_suit=None, tag=None, selected=False, dim=False):
    """Draw a single card centered at (cx, cy). All pieces share `tag`
    (plus always the group tag 'tablecards' if the caller includes it)."""
    tags = tuple(tag) if isinstance(tag, (list, tuple)) else (tag,) if tag else ()
    y_lift = -8 if selected else 0
    x1, y1 = cx - w / 2, cy - h / 2 + y_lift
    x2, y2 = cx + w / 2, cy + h / 2 + y_lift

    if face_down:
        rounded_rect(canvas, x1, y1, x2, y2, r=8, fill="#2b5fa8",
                     outline="black", width=2, tags=tags)
        canvas.create_rectangle(x1 + 5, y1 + 5, x2 - 5, y2 - 5,
                                 outline="#9fc1e8", width=1.5, tags=tags)
        return

    is_trump = trump_suit is not None and card.suit == trump_suit
    fill = "#dddddd" if dim else (TRUMP_FILL if is_trump else "white")
    outline = "#1a7a1a" if selected else "black"
    width_ = 3 if selected else 2
    rounded_rect(canvas, x1, y1, x2, y2, r=10, fill=fill, outline=outline,
                 width=width_, tags=tags)

    color = "#888888" if dim else ("red" if card.suit in RED_SUITS else "black")
    canvas.create_text(x1 + 10, y1 + 14, text=str(card), anchor="w",
                        font=("Consolas", 11, "bold"), fill=color, tags=tags)
    canvas.create_text(x2 - 10, y2 - 14, text=str(card), anchor="e",
                        font=("Consolas", 11, "bold"), fill=color, tags=tags)
    canvas.create_text(cx, cy + y_lift, text=SUIT_SYMBOLS.get(card.suit, card.suit),
                        font=("Consolas", 26), fill=color, tags=tags)


class TanTableGUI:
    def __init__(self, env, human_seat: int = 0, agents: dict | None = None,
                 ai_delay_ms: int = 700):
        self.env = env
        self.human_seat = human_seat
        self.agents = agents or {}
        self.ai_delay_ms = ai_delay_ms

        self.selected_attack_card = None
        self.state = None
        self.player_id = None
        self.seat_pos: dict[int, tuple[float, float]] = {}
        self.table_center: tuple[float, float] = (0, 0)

        # Fixed spot on the LEFT side of the table (no seat lives here) that
        # represents the discard pile. Successful rounds animate to here.
        self.discard_pos = (70, CANVAS_H * 0.47)

        self.root = tk.Tk()
        self.root.title("Tan")
        self._build_layout()
        self.new_game()
        self.root.mainloop()

    # ------------------------------------------------------------------ #
    def _build_layout(self):
        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Arial", 13, "bold")).pack(pady=(8, 2))

        self.info_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.info_var,
                 font=("Arial", 10)).pack()

        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H,
                                 bg=TABLE_BG, highlightthickness=0)
        self.canvas.pack(pady=6)

        self.action_frame = tk.Frame(self.root)
        self.action_frame.pack(pady=4)

        self.log = tk.Text(self.root, height=6, width=80, state="disabled")
        self.log.pack(pady=(4, 10))

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ------------------------------------------------------------------ #
    # game loop
    # ------------------------------------------------------------------ #
    def new_game(self):
        self.state, self.player_id = self.env.reset()
        self.selected_attack_card = None
        self.seat_pos = {}
        self._log("=== New game dealt ===")
        self.render()
        self.root.after(50, self.advance)

    def advance(self):
        g = self.env.game
        if g.is_over():
            self.render()
            self.root.after(300, self.show_results)
            return

        if self.player_id == self.human_seat:
            self.render()
            return

        actor = self.player_id
        agent = self.agents.get(actor)
        if agent is None:
            raise RuntimeError(f"No agent registered for seat {actor}")

        action_id = self._agent_action(agent)
        action = id_to_action(action_id, actor)
        self._log(f"Player {actor}: {self._describe(action)}")
        self._do_step(action, self.ai_delay_ms)

    def _agent_action(self, agent) -> int:
        if hasattr(agent, "eval_step"):
            action_id, _ = agent.eval_step(self.state)
        else:
            action_id = agent.step(self.state)
        return int(action_id)

    def _play(self, action):
        self._log(f"You: {self._describe(action)}")
        self.selected_attack_card = None
        self._do_step(action, self.ai_delay_ms)

    def _do_step(self, action, delay):
        """Steps the env and detects whether THIS step is the one that
        sweeps the table away -- either into the defender's hand (a real
        pickup) or into the discard pile (a successful round).

        We key this directly off the *defender*, whose identity we already
        know before stepping, rather than scanning every player's hand size
        for "who grew" -- that approach is ambiguous because, after a round
        ends (win OR loss), several players may draw replenishment cards
        from the draw pile in the very same step. Scanning players in seat
        order would latch onto whichever player happens to be checked
        first (often the human), sending the animation to the wrong place.
        """
        g = self.env.game.game
        defender_before = g.defender
        pile_nonempty_before = bool(g.attack_pile)
        defender_hand_before = len(g.hands[defender_before])

        self.state, self.player_id = self.env.step(action_to_id(action))

        g2 = self.env.game.game
        pile_nonempty_after = bool(g2.attack_pile)

        if pile_nonempty_before and not pile_nonempty_after:
            picked_up = len(g2.hands[defender_before]) > defender_hand_before
            if picked_up:
                dest = self.seat_pos.get(defender_before)
            else:
                dest = self.discard_pos  # successful defense -> discard pile
            if dest is not None:
                self._animate_take(dest, lambda: self._post_step(delay))
                return

        self._post_step(delay)

    def _post_step(self, delay):
        self.render()
        self.root.after(delay, self.advance)

    def _animate_take(self, dest, callback, steps=15, delay=18):
        sx, sy = self.table_center
        dx = (dest[0] - sx) / steps
        dy = (dest[1] - sy) / steps

        def step(i):
            if i >= steps:
                callback()
                return
            self.canvas.move("tablecards", dx, dy)
            self.root.after(delay, lambda: step(i + 1))

        step(0)

    def _select_attack_card(self, card):
        self.selected_attack_card = card
        self.render()

    @staticmethod
    def _describe(action) -> str:
        if isinstance(action, Attack):
            return f"attacks with {action.card}"
        if isinstance(action, Defend):
            return f"defends {action.attacking_card} with {action.card}"
        if isinstance(action, Throwin):
            return f"throws in {action.card}"
        if isinstance(action, AttackPass):
            return "passes"
        if isinstance(action, DefendPass):
            return "takes the cards"
        if isinstance(action, Continue):
            return "continues"
        if isinstance(action, SwapTrump):
            return "swaps the trump card"
        if isinstance(action, DeclineSwap):
            return "keeps their hand (declines swap)"
        return str(action)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def _card_action_for(self, card, legal):
        for a in legal:
            if isinstance(a, Attack) and a.card == card:
                return a
            if isinstance(a, Throwin) and a.card == card:
                return a
            if (isinstance(a, Defend) and a.card == card
                    and a.attacking_card == self.selected_attack_card):
                return a
        return None

    def render(self):
        g = self.env.game.game
        legal = self.env.game.legal
        cur = self.env.game.current_player
        n = self.env.num_players
        trump_suit = g.trump.suit

        self.canvas.delete("all")
        cw, ch = CANVAS_W, CANVAS_H

        # status -------------------------------------------------------
        if g.game_over:
            self.status_var.set("Game finished!")
        elif cur == self.human_seat:
            self.status_var.set("Your turn")
        else:
            self.status_var.set(f"Player {cur}'s turn...")

        self.info_var.set(
            f"Draw pile: {len(g.draw_pile)}   "
            f"Attacker: P{g.current_attacker}   Defender: P{g.defender}"
        )

        # trump card (drawn as a real card, top-left corner) -----------
        trump_x, trump_y = 55, 65
        self.canvas.create_text(trump_x, trump_y - 55, text="Trump",
                                 fill="white", font=("Arial", 10, "bold"))
        draw_card(self.canvas, trump_x, trump_y, card=g.trump,
                  trump_suit=trump_suit, tag="trumpcard")

        # discard pile (fixed spot on the left, no seat lives here) -----
        dx0, dy0 = self.discard_pos
        self.canvas.create_text(dx0, dy0 - CARD_H / 2 - 16, text="Discard",
                                 fill="white", font=("Arial", 10, "bold"))
        n_discard = len(g.discard_pile)
        if n_discard:
            stack_depth = min(n_discard, 6)
            for k in range(stack_depth):
                draw_card(self.canvas, dx0 + k * 2, dy0 + k * 2, face_down=True,
                          w=MINI_W + 10, h=MINI_H + 14, tag=f"discard_{k}")
            self.canvas.create_text(dx0, dy0 + MINI_H / 2 + 24,
                                     text=str(n_discard), fill="white",
                                     font=("Arial", 10, "bold"))
        else:
            rounded_rect(self.canvas, dx0 - (MINI_W + 10) / 2, dy0 - (MINI_H + 14) / 2,
                         dx0 + (MINI_W + 10) / 2, dy0 + (MINI_H + 14) / 2,
                         r=8, outline="#88c49a", width=2, fill="", dash=(4, 3))

        # opponents ------------------------------------------------------
        opponents = [p for p in range(n) if p != self.human_seat]
        opp_y = 70
        if opponents:
            step = min(170, (cw - 160) / len(opponents))
            start_x = (cw - step * (len(opponents) - 1)) / 2 if len(opponents) > 1 else cw / 2
        for i, p in enumerate(opponents):
            cx = start_x + i * step if opponents else cw / 2
            count = len(g.hands[p])
            fan_dx = 14
            for k in range(count):
                draw_card(self.canvas, cx + k * fan_dx, opp_y, face_down=True,
                          w=MINI_W, h=MINI_H, tag=f"opp_{p}_{k}")
            self.seat_pos[p] = (cx + (count - 1) * fan_dx / 2, opp_y)

            role = []
            if p == g.defender: role.append("DEFENDER")
            if p == g.current_attacker: role.append("MAIN ATK")
            if g.passed[p]: role.append("passed")
            if not g.current_players[p]: role.append("out")
            label = f"P{p}  ({count}){'  ' + ','.join(role) if role else ''}"
            self.canvas.create_text(cx, opp_y + MINI_H / 2 + 16, text=label,
                                     fill="white", font=("Arial", 10, "bold"))

        # table / attack pile ---------------------------------------------
        table_y = ch * 0.47
        defendable_atks = {a.attacking_card for a in legal if isinstance(a, Defend)}
        if cur == self.human_seat:
            if len(defendable_atks) == 1:
                self.selected_attack_card = next(iter(defendable_atks))
            elif self.selected_attack_card not in defendable_atks:
                self.selected_attack_card = None

        pile = g.attack_pile
        if pile:
            gap = CARD_W + 20
            start_x = (cw - gap * (len(pile) - 1)) / 2
            centers = []
            for idx, (atk, dfn) in enumerate(pile):
                cx = start_x + idx * gap
                centers.append(cx)
                selectable = (cur == self.human_seat and dfn is None
                              and atk in defendable_atks)
                atk_tag = f"atk_{idx}"
                draw_card(self.canvas, cx, table_y, card=atk,
                          trump_suit=trump_suit, tag=[atk_tag, "tablecards"],
                          selected=(atk == self.selected_attack_card))
                if selectable:
                    self.canvas.tag_bind(
                        atk_tag, "<Button-1>",
                        lambda e, c=atk: self._select_attack_card(c))
                if dfn is not None:
                    draw_card(self.canvas, cx + 20, table_y + 20, card=dfn,
                              trump_suit=trump_suit,
                              tag=[f"def_{idx}", "tablecards"])
            self.table_center = (sum(centers) / len(centers), table_y)
        else:
            self.canvas.create_text(cw / 2, table_y, text="(table empty)",
                                     fill="white", font=("Arial", 12))
            self.table_center = (cw / 2, table_y)

        # action buttons ---------------------------------------------------
        for w in self.action_frame.winfo_children():
            w.destroy()
        if cur == self.human_seat:
            labels = {
                AttackPass: "Pass",
                DefendPass: "Take cards",
                Continue: "Continue",
                DeclineSwap: "Keep hand",
            }
            for a in legal:
                if isinstance(a, SwapTrump):
                    text = f"Swap trump ({g.trump})"
                elif type(a) in labels:
                    text = labels[type(a)]
                else:
                    continue
                tk.Button(self.action_frame, text=text, width=16,
                          command=lambda a=a: self._play(a)).pack(side="left", padx=4)

        # human hand ---------------------------------------------------
        hand_y = ch - 90
        hand = sorted(g.hands[self.human_seat],
                      key=lambda c: (c.suit, RANK_VALUE[c.rank]))
        self.seat_pos[self.human_seat] = (cw / 2, hand_y)
        if hand:
            step = min(CARD_W + 15, (cw - 80) / len(hand))
            start_x = (cw - step * (len(hand) - 1)) / 2
            for i, c in enumerate(hand):
                cx = start_x + i * step
                action = self._card_action_for(c, legal) if cur == self.human_seat else None
                dim = (cur == self.human_seat and action is None)
                tag = f"hand_{c.suit}{c.rank}"
                draw_card(self.canvas, cx, hand_y, card=c, trump_suit=trump_suit,
                          tag=tag, dim=dim)
                if action is not None:
                    self.canvas.tag_bind(
                        tag, "<Button-1>", lambda e, a=action: self._play(a))

    # ------------------------------------------------------------------ #
    def show_results(self):
        payoffs = self.env.get_payoffs()
        msg = "\n".join(f"Player {p}: payoff {payoffs[p]:+.2f}"
                         for p in range(len(payoffs)))
        if messagebox.askyesno("Game over", msg + "\n\nPlay again?"):
            self.new_game()
        else:
            self.root.destroy()


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    import threading
    import rlcard
    import tan_rlcard  # noqa: F401  (registers the "tan" env with rlcard.make)

    NUM_PLAYERS = 3
    HUMAN_SEAT = 0

    # Show a loading window immediately so the user knows the app started.
    splash = tk.Tk()
    splash.title("Tan")
    splash.resizable(False, False)
    splash_w, splash_h = 320, 100
    sw, sh = splash.winfo_screenwidth(), splash.winfo_screenheight()
    splash.geometry(f"{splash_w}x{splash_h}+{(sw - splash_w) // 2}+{(sh - splash_h) // 2}")
    tk.Label(splash, text="Loading Tan…", font=("Arial", 16, "bold")).pack(expand=True)
    tk.Label(splash, text="Please wait", font=("Arial", 11)).pack(pady=(0, 12))
    splash.update()

    def load_in_background():
        try:
            env = rlcard.make("tan", config={"game_num_players": NUM_PLAYERS})
            agents = {
                seat: create_dmc_agent(seat)
                for seat in range(NUM_PLAYERS) if seat != HUMAN_SEAT
            }
            splash.after(0, lambda: launch(env, agents))
        except BaseException as exc:
            import traceback
            msg = traceback.format_exc()
            splash.after(0, lambda: show_load_error(msg))

    def launch(env, agents):
        try:
            splash.destroy()
            TanTableGUI(env, human_seat=HUMAN_SEAT, agents=agents)
        except BaseException as exc:
            import traceback
            msg = traceback.format_exc()
            show_load_error(msg)

    def show_load_error(msg):
        try:
            splash.destroy()
        except Exception:
            pass
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror("Failed to load", msg)
        root.destroy()

    # Load on a background thread so the splash window stays responsive.
    t = threading.Thread(target=load_in_background, daemon=True)
    t.start()
    splash.mainloop()