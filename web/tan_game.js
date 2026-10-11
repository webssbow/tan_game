// tan_game.js -- port of tan_game.py + tan_rlcard.py observation encoder

// -------------------------------------------------------------------------
// Cards
// -------------------------------------------------------------------------
export const ALL_RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
export const ALL_SUITS = ['D','C','H','S'];
export const RANK_VALUE = Object.fromEntries(ALL_RANKS.map((r,i) => [r,i]));
export const SUIT_SYMBOLS = {D:'♦', C:'♣', H:'♥', S:'♠'};
const VALID_TRUMP_RANKS = new Set('3456789');

export function cardStr(c) { return c.rank + SUIT_SYMBOLS[c.suit]; }

// Use plain objects {suit, rank} as cards; intern them so === works
const _CARD_POOL = {};
export function Card(suit, rank) {
  const k = suit + rank;
  if (!_CARD_POOL[k]) _CARD_POOL[k] = Object.freeze({suit, rank});
  return _CARD_POOL[k];
}

// All 52 cards in the fixed order used by the RL encoding
export const ALL_CARDS = [];
for (const rank of ALL_RANKS) for (const suit of ALL_SUITS) ALL_CARDS.push(Card(suit, rank));
export const CARD_ID = new Map(ALL_CARDS.map((c,i) => [c,i]));

// -------------------------------------------------------------------------
// Actions (plain tagged objects; frozen for safety)
// -------------------------------------------------------------------------
export const AT = Object.freeze({
  ATTACK: 'Attack', DEFEND: 'Defend', ATTACK_PASS: 'AttackPass',
  DEFEND_PASS: 'DefendPass', THROWIN: 'Throwin', CONTINUE: 'Continue',
  SWAP_TRUMP: 'SwapTrump', DECLINE_SWAP: 'DeclineSwap',
});

export function mkAttack(player, card)              { return {t:AT.ATTACK,      player, card}; }
export function mkDefend(player, atkCard, card)     { return {t:AT.DEFEND,      player, atkCard, card}; }
export function mkAttackPass(player)                { return {t:AT.ATTACK_PASS, player}; }
export function mkDefendPass(player)                { return {t:AT.DEFEND_PASS, player}; }
export function mkThrowin(player, card)             { return {t:AT.THROWIN,     player, card}; }
export function mkContinue(player)                  { return {t:AT.CONTINUE,    player}; }
export function mkSwapTrump(player)                 { return {t:AT.SWAP_TRUMP,  player}; }
export function mkDeclineSwap(player)               { return {t:AT.DECLINE_SWAP, player}; }

// -------------------------------------------------------------------------
// Action <-> id encoding (mirrors tan_rlcard.py)
// -------------------------------------------------------------------------
const ATTACK_BASE    = 0;
const THROWIN_BASE   = 52;
const DEFEND_BASE    = 104;
const ATTACK_PASS_ID = DEFEND_BASE + 52 * 52; // 2808
const DEFEND_PASS_ID = ATTACK_PASS_ID + 1;
const CONTINUE_ID    = ATTACK_PASS_ID + 2;
const SWAP_TRUMP_ID  = ATTACK_PASS_ID + 3;
const DECLINE_SWAP_ID= ATTACK_PASS_ID + 4;
export const NUM_ACTIONS = ATTACK_PASS_ID + 5; // 2813

export function actionToId(a) {
  switch (a.t) {
    case AT.ATTACK:      return ATTACK_BASE    + CARD_ID.get(a.card);
    case AT.THROWIN:     return THROWIN_BASE   + CARD_ID.get(a.card);
    case AT.DEFEND:      return DEFEND_BASE    + 52*CARD_ID.get(a.atkCard) + CARD_ID.get(a.card);
    case AT.ATTACK_PASS: return ATTACK_PASS_ID;
    case AT.DEFEND_PASS: return DEFEND_PASS_ID;
    case AT.CONTINUE:    return CONTINUE_ID;
    case AT.SWAP_TRUMP:  return SWAP_TRUMP_ID;
    case AT.DECLINE_SWAP:return DECLINE_SWAP_ID;
    default: throw new Error(`unknown action ${a.t}`);
  }
}

export function idToAction(aid, player) {
  if (aid < THROWIN_BASE)   return mkAttack(player, ALL_CARDS[aid - ATTACK_BASE]);
  if (aid < DEFEND_BASE)    return mkThrowin(player, ALL_CARDS[aid - THROWIN_BASE]);
  if (aid < ATTACK_PASS_ID) {
    const k = aid - DEFEND_BASE;
    return mkDefend(player, ALL_CARDS[Math.floor(k/52)], ALL_CARDS[k%52]);
  }
  const m = {[ATTACK_PASS_ID]: mkAttackPass, [DEFEND_PASS_ID]: mkDefendPass,
             [CONTINUE_ID]: mkContinue, [SWAP_TRUMP_ID]: mkSwapTrump,
             [DECLINE_SWAP_ID]: mkDeclineSwap};
  return m[aid](player);
}

// -------------------------------------------------------------------------
// Seeded PRNG (xoshiro128** — gives reproducible games)
// -------------------------------------------------------------------------
function Rng(seed) {
  let s = new Uint32Array(4);
  // splitmix64 to initialise
  let x = BigInt(seed >>> 0);
  for (let i=0; i<4; i++) {
    x += 0x9e3779b97f4a7c15n;
    let z = x;
    z = (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n;
    z = (z ^ (z >> 27n)) * 0x94d049bb133111ebn;
    z = z ^ (z >> 31n);
    s[i] = Number(BigInt.asUintN(32, z));
  }
  function rotl(x,k) { return ((x << k) | (x >>> (32-k))) >>> 0; }
  function next() {
    const result = Math.imul(rotl(Math.imul(s[1],5)>>>0,7),9)>>>0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3],11);
    return result;
  }
  return {
    shuffle(arr) {
      for (let i=arr.length-1; i>0; i--) {
        const j = Math.floor((next()/4294967296) * (i+1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
    },
    int(n) { return Math.floor((next()/4294967296) * n); }
  };
}

// -------------------------------------------------------------------------
// TanGame (direct port of tan_game.py)
// -------------------------------------------------------------------------
function validAttack(c1, c2, trumpSuit) {
  const comp = c2.suit === c1.suit && RANK_VALUE[c2.rank] > RANK_VALUE[c1.rank];
  if (c1.suit === trumpSuit) return comp;
  return c2.suit === trumpSuit || comp;
}

export class TanGame {
  constructor(numPlayers, handSize, rng) {
    this.numPlayers = numPlayers;
    this.handSize = handSize;
    this.rng = rng;
    this.startGame();
  }

  startGame() {
    this.hands = Array.from({length:this.numPlayers}, () => []);
    this.drawPile = [...ALL_CARDS];
    this.discardPile = [];
    this.roundStarted = false;
    this.attackPile = [];   // [{atk:Card, dfn:Card|null}]
    this.passed = new Array(this.numPlayers).fill(false);
    this.canThrowin = false;
    this.currentPlayers = new Array(this.numPlayers).fill(true);
    this.finishedPlayers = [];
    this.freeAttackPile = false;
    this.currentAttacker = 0;
    this.defender = 1;

    let dealt = false;
    while (!dealt) {
      this.rng.shuffle(this.drawPile);
      for (let r=0; r<this.handSize; r++)
        for (let p=0; p<this.numPlayers; p++)
          this.hands[p].push(this.drawPile.pop());
      if (this.drawPile.some(c => VALID_TRUMP_RANKS.has(c.rank))) {
        dealt = true;
      } else {
        for (const hand of this.hands) { this.drawPile.push(...hand); hand.length=0; }
      }
    }
    while (true) {
      this.trump = this.drawPile[this.drawPile.length-1];
      if (VALID_TRUMP_RANKS.has(this.trump.rank)) {
        this.drawPile.pop();
        this.drawPile.unshift(this.trump);
        break;
      }
      this.rng.shuffle(this.drawPile);
    }
  }

  get gameOver() { return this.currentPlayers.filter(Boolean).length <= 1; }

  attackers() { return [...Array(this.numPlayers).keys()].filter(p => this.currentPlayers[p] && p !== this.defender); }

  nextActive(seat) {
    const N = this.numPlayers;
    for (let k=1; k<=N; k++) {
      const s = (seat+k) % N;
      if (this.currentPlayers[s]) return s;
    }
    throw new Error('no active player found');
  }

  finish(p) {
    this.currentPlayers[p] = false;
    this.passed[p] = true;
    this.finishedPlayers.push(p);
  }

  refill(player) {
    while (this.drawPile.length > 0 && this.hands[player].length < this.handSize)
      this.hands[player].push(this.drawPile.pop());
  }

  refillOrDeclareWin(player) {
    if (this.drawPile.length > 0) { this.refill(player); return; }
    this.finish(player);
    if (player === this.defender) {
      this.freeAttackPile = true;
      for (const p of this.attackers()) this.passed[p] = false;
    }
  }

  advanceRound(defenderWon) {
    const cards = this.attackPile.flatMap(({atk,dfn}) => dfn ? [atk,dfn] : [atk]);
    if (defenderWon) this.discardPile.push(...cards);
    else this.hands[this.defender].push(...cards);
    this.attackPile = [];

    const n = this.numPlayers;
    const order = [this.currentAttacker, this.defender];
    for (let k=1; k<n; k++) {
      const s = (this.defender+k)%n;
      if (!order.includes(s)) order.push(s);
    }
    for (const p of order) if (this.currentPlayers[p]) this.refill(p);
    for (let p=0; p<n; p++) if (this.currentPlayers[p] && this.hands[p].length===0) this.finish(p);

    this.passed = this.currentPlayers.map(a => !a);
    this.canThrowin = false;
    this.freeAttackPile = false;
    if (this.gameOver) return;

    let nxt;
    if (defenderWon && this.currentPlayers[this.defender]) nxt = this.defender;
    else nxt = this.nextActive(this.defender);
    this.currentAttacker = nxt;
    this.defender = this.nextActive(nxt);
  }

  validActions() {
    if (this.gameOver) return [];
    const valid = [];

    if (!this.roundStarted) {
      for (let i=0; i<this.numPlayers; i++) {
        if (this.hands[i].some(c => c.suit===this.trump.suit && c.rank==='2')) {
          valid.push(mkSwapTrump(i)); break;
        }
      }
    }

    const pileRanks = () => new Set(this.attackPile.flatMap(({atk,dfn}) => dfn?[atk.rank,dfn.rank]:[atk.rank]));

    if (!this.attackPile.length) {
      for (const card of this.hands[this.currentAttacker])
        valid.push(mkAttack(this.currentAttacker, card));
    } else if (this.freeAttackPile) {
      const ranks = pileRanks();
      for (let i=0; i<this.numPlayers; i++) {
        if (!this.passed[i]) valid.push(mkAttackPass(i));
        for (const card of this.hands[i])
          if (ranks.has(card.rank)) valid.push(mkAttack(i, card));
      }
    } else if (this.canThrowin) {
      const ranks = pileRanks();
      for (const card of this.hands[this.defender])
        if (ranks.has(card.rank)) valid.push(mkThrowin(this.defender, card));
      valid.push(mkContinue(this.defender));
    } else {
      for (let i=0; i<this.numPlayers; i++) {
        if (i === this.defender) {
          if (this.passed[i]) continue;
          if (!this.attackPile.every(({dfn}) => dfn !== null))
            valid.push(mkDefendPass(i));
          for (const {atk,dfn} of this.attackPile) {
            if (dfn !== null) continue;
            for (const card of this.hands[i])
              if (validAttack(atk, card, this.trump.suit))
                valid.push(mkDefend(i, atk, card));
          }
        } else {
          if (!this.passed[i]) valid.push(mkAttackPass(i));
          const ranks = pileRanks();
          for (const card of this.hands[i])
            if (ranks.has(card.rank)) valid.push(mkAttack(i, card));
        }
      }
    }
    return valid;
  }

  processAction(a) {
    switch (a.t) {
      case AT.ATTACK: {
        this.passed[a.player] = false;
        const h = this.hands[a.player];
        h.splice(h.indexOf(a.card), 1);
        this.attackPile.push({atk:a.card, dfn:null});
        if (!h.length) this.refillOrDeclareWin(a.player);
        break;
      }
      case AT.DEFEND: {
        const h = this.hands[this.defender];
        h.splice(h.indexOf(a.card), 1);
        const pileRanks = new Set(this.attackPile.flatMap(({atk,dfn})=>dfn?[atk.rank,dfn.rank]:[atk.rank]));
        if (!pileRanks.has(a.card.rank))
          this.passed = this.currentPlayers.map(v => !v);
        const idx = this.attackPile.findIndex(p => p.atk === a.atkCard);
        this.attackPile[idx] = {atk:a.atkCard, dfn:a.card};
        if (!h.length) this.refillOrDeclareWin(this.defender);
        break;
      }
      case AT.ATTACK_PASS:
        this.passed[a.player] = true; break;
      case AT.DEFEND_PASS:
        this.passed[this.defender] = true;
        for (const p of this.attackers()) this.passed[p] = false;
        break;
      case AT.THROWIN: {
        const h = this.hands[this.defender];
        h.splice(h.indexOf(a.card), 1);
        this.attackPile.push({atk:a.card, dfn:null});
        break;
      }
      case AT.CONTINUE:
        this.advanceRound(true); break;
      case AT.SWAP_TRUMP: {
        const trump2 = Card(this.trump.suit, '2');
        const h = this.hands[a.player];
        h.splice(h.indexOf(trump2), 1);
        h.push(this.trump);
        this.trump = trump2;
        this.drawPile[0] = trump2;
        this.currentAttacker = a.player;
        this.defender = this.nextActive(a.player);
        break;
      }
      default: throw new Error(`unknown action ${a.t}`);
    }

    if (this.gameOver || !this.attackPile.length) return;
    const atkDone = this.attackers().every(p => this.passed[p]);
    if (this.freeAttackPile) {
      if (atkDone) this.advanceRound(true);
    } else if (this.passed[this.defender]) {
      if (atkDone) this.advanceRound(false);
    } else if (atkDone && this.attackPile.every(({dfn}) => dfn !== null)) {
      this.canThrowin = true;
    }
    this.roundStarted = true;
  }
}

// -------------------------------------------------------------------------
// TanRLGame adapter (mirrors tan_rlcard.py TanRLGame)
// -------------------------------------------------------------------------
export class TanRLGame {
  constructor({numPlayers=3, handSize=6, defenderFirst=true, attackerOrder='main_first', seed=0}={}) {
    this.numPlayers = numPlayers;
    this.handSize = handSize;
    this.defenderFirst = defenderFirst;
    this.attackerOrder = attackerOrder;
    this.seed = seed;
    this.game = null;
    this.currentPlayer = 0;
    this.legal = [];
    this.lastActor = 0;
    this.swapDeclined = false;
  }

  initGame() {
    this.game = new TanGame(this.numPlayers, this.handSize, Rng(this.seed + Math.random()*1e9|0));
    this.swapDeclined = false;
    this.lastActor = this.game.currentAttacker;
    this._arbitrate();
    return this.currentPlayer;
  }

  step(action) {
    if (action.t === AT.DECLINE_SWAP) {
      this.swapDeclined = true;
    } else {
      this.game.processAction(action);
    }
    this.lastActor = action.player;
    this._arbitrate();
  }

  _arbitrate() {
    const g = this.game;
    if (g.gameOver) { this.legal = []; return; }

    const valid = g.validActions();
    const swap = valid.find(a => a.t === AT.SWAP_TRUMP);
    if (swap && !this.swapDeclined) {
      this.currentPlayer = swap.player;
      this.legal = [swap, mkDeclineSwap(swap.player)];
      return;
    }

    const byPlayer = new Map();
    for (const a of valid) {
      if (a.t === AT.SWAP_TRUMP) continue;
      if (!byPlayer.has(a.player)) byPlayer.set(a.player, []);
      byPlayer.get(a.player).push(a);
    }

    if (this.defenderFirst && byPlayer.has(g.defender)) {
      this.currentPlayer = g.defender;
      this.legal = byPlayer.get(g.defender);
      return;
    }

    const candidates = [...byPlayer.keys()].filter(p => p !== g.defender && !g.passed[p]);
    if (candidates.length) {
      this.currentPlayer = this._pickAttacker(candidates);
      this.legal = byPlayer.get(this.currentPlayer);
      return;
    }

    if (byPlayer.has(g.defender)) {
      this.currentPlayer = g.defender;
      this.legal = byPlayer.get(g.defender);
      return;
    }

    throw new Error('nobody can act but game is not over');
  }

  _pickAttacker(candidates) {
    const n = this.numPlayers;
    if (this.attackerOrder === 'random') return candidates[Math.floor(Math.random()*candidates.length)];
    const start = this.attackerOrder === 'main_first' ? this.game.currentAttacker : (this.lastActor+1)%n;
    for (let k=0; k<n; k++) { const p=(start+k)%n; if (candidates.includes(p)) return p; }
    throw new Error('unreachable');
  }

  // Build the observation vector (mirrors TanEnv._encode_obs)
  encodeObs(seat) {
    const g = this.game;
    const n = this.numPlayers;
    const obs = new Float32Array(6*52 + 4 + 5*n + 4);
    let off = 0;

    // plane 0: my hand
    for (const c of g.hands[seat]) obs[off + CARD_ID.get(c)] = 1;
    off += 52;
    // planes 1-3: attack pile
    for (const {atk,dfn} of g.attackPile) {
      if (dfn === null) { obs[off + CARD_ID.get(atk)] = 1; }
      else {
        obs[off+52 + CARD_ID.get(atk)] = 1;
        obs[off+104 + CARD_ID.get(dfn)] = 1;
      }
    }
    off += 52*3;
    // plane 4: discard
    for (const c of g.discardPile) obs[off + CARD_ID.get(c)] = 1;
    off += 52;
    // plane 5: trump card
    obs[off + CARD_ID.get(g.trump)] = 1;
    off += 52;
    // trump suit one-hot
    obs[off + ALL_SUITS.indexOf(g.trump.suit)] = 1;
    off += 4;
    // seats (relative, row 0 = me)
    for (let k=0; k<n; k++) {
      const p = (seat+k)%n;
      obs[off + k*5 + 0] = g.hands[p].length / this.handSize;
      obs[off + k*5 + 1] = p === g.defender ? 1 : 0;
      obs[off + k*5 + 2] = p === g.currentAttacker ? 1 : 0;
      obs[off + k*5 + 3] = g.passed[p] ? 1 : 0;
      obs[off + k*5 + 4] = g.currentPlayers[p] ? 1 : 0;
    }
    off += 5*n;
    // misc
    obs[off+0] = g.drawPile.length / 52;
    obs[off+1] = g.freeAttackPile ? 1 : 0;
    obs[off+2] = g.canThrowin ? 1 : 0;
    obs[off+3] = this.legal.some(a => a.t === AT.SWAP_TRUMP) ? 1 : 0;
    return obs;
  }

  payoffs() {
    const g = this.game;
    const n = this.numPlayers;
    const order = [...g.finishedPlayers, ...[...Array(n).keys()].filter(p => g.currentPlayers[p])];
    const p = new Float32Array(n);
    for (let rank=0; rank<order.length; rank++) p[order[rank]] = 1 - 2*rank/(n-1);
    return p;
  }
}
