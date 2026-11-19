"""
Refactored SPX options GEX/DEX/VEX/flow analyzer.


Notes:
- GEX is an estimated public-market convention, not observed dealer positioning.
- DEX is raw option OI delta, not dealer delta.
- VEX is gross call/put vega, not signed dealer vega.
- Bid/ask flow is an aggressor-side heuristic; it cannot determine open/close.
"""


from __future__ import annotations


from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
import math
import time


CONTRACT_MULTIPLIER = 100
GEX_IV_SCALE = 0.01




def num(x: Any, default: float = 0.0) -> float:
   if x is None or x == "":
       return default
   try:
       return float(x)
   except (TypeError, ValueError):
       return default




def integer(x: Any, default: int = 0) -> int:
   if x is None or x == "":
       return default
   try:
       return int(float(x))
   except (TypeError, ValueError):
       return default




# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


def classify_aggressor(
   last: Any,
   bid: Any,
   ask: Any,
   tolerance: float = 0.01,
) -> str:
   """Estimate whether a trade hit the bid or lifted the ask."""
   last, bid, ask = num(last, math.nan), num(bid, math.nan), num(ask, math.nan)


   if not all(math.isfinite(x) for x in (last, bid, ask)):
       return "UNKNOWN"
   if bid < 0 or ask < bid:
       return "UNKNOWN"


   if ask == bid:
       if abs(last - ask) <= tolerance:
           return "UNKNOWN"
       return "BUY" if last > ask else "SELL"


   spread = ask - bid
   tol = max(tolerance, spread * 0.10)


   if last >= ask - tol:
       return "BUY"
   if last <= bid + tol:
       return "SELL"
   return "UNKNOWN"




def compute_flow(
   chain: Iterable[Dict[str, Any]],
   tolerance: float = 0.01,
) -> Dict[str, Any]:
   """
   Estimate aggressor flow.


   Positive directional score = more bullish pressure.
   Negative directional score = more bearish pressure.


   This is a heuristic, not a dealer/opening-flow detector.
   """
   totals = {
       "call_buy": 0,
       "call_sell": 0,
       "call_unknown": 0,
       "put_buy": 0,
       "put_sell": 0,
       "put_unknown": 0,
   }


   strike_flow = defaultdict(
       lambda: {"call_buy": 0, "call_sell": 0, "put_buy": 0, "put_sell": 0}
   )


   for opt in chain:
       strike = opt.get("strike")
       typ = str(opt.get("option_type", "")).lower()
       volume = integer(opt.get("volume"))


       if strike is None or typ not in ("call", "put") or volume <= 0:
           continue


       side = classify_aggressor(
           opt.get("last"), opt.get("bid"), opt.get("ask"), tolerance
       )
       strike = num(strike)


       if typ == "call":
           if side == "BUY":
               totals["call_buy"] += volume
               strike_flow[strike]["call_buy"] += volume
           elif side == "SELL":
               totals["call_sell"] += volume
               strike_flow[strike]["call_sell"] += volume
           else:
               totals["call_unknown"] += volume
       else:
           if side == "BUY":
               totals["put_buy"] += volume
               strike_flow[strike]["put_buy"] += volume
           elif side == "SELL":
               totals["put_sell"] += volume
               strike_flow[strike]["put_sell"] += volume
           else:
               totals["put_unknown"] += volume


   classified = (
       totals["call_buy"] + totals["call_sell"]
       + totals["put_buy"] + totals["put_sell"]
   )
   all_volume = classified + totals["call_unknown"] + totals["put_unknown"]


   score = (
       totals["call_buy"]
       - 0.5 * totals["call_sell"]
       - totals["put_buy"]
       + 0.5 * totals["put_sell"]
   )


   return {
       "totals": totals,
       "directional_score": score,
       "directional_score_pct": score / classified * 100 if classified else 0.0,
       "classification_coverage_pct": classified / all_volume * 100 if all_volume else 0.0,
       "strike_flow": dict(strike_flow),
   }


def compute_live_flow(
    previous: Dict[str, int],
    chain: Iterable[Dict[str, Any]],
    tolerance: float = 0.01,
) -> Dict[str, Any]:
   """Estimate new option flow from volume increments between chain snapshots.

   Tradier's chain endpoint is a quote/volume snapshot, not an OPRA trade tape.
   The returned prints therefore represent newly observed volume only.
   """
   events = []
   totals = {"call_buy": 0, "call_sell": 0, "call_unknown": 0,
             "put_buy": 0, "put_sell": 0, "put_unknown": 0}

   for opt in chain:
       strike = opt.get("strike")
       option_type = str(opt.get("option_type", "")).lower()
       symbol = opt.get("symbol") or f"{option_type}:{strike}"
       current_volume = integer(opt.get("volume"))
       prior_volume = integer(previous.get(symbol))
       delta_volume = max(0, current_volume - prior_volume)
       if strike is None or option_type not in ("call", "put") or delta_volume <= 0:
           continue

       side = classify_aggressor(opt.get("last"), opt.get("bid"), opt.get("ask"), tolerance)
       bucket = f"{option_type}_{side.lower()}" if side in ("BUY", "SELL") else f"{option_type}_unknown"
       totals[bucket] += delta_volume
       events.append({
           "symbol": symbol,
           "strike": num(strike),
           "option_type": option_type,
           "expiration": opt.get("expiration_date") or opt.get("expiration"),
           "dte": opt.get("dte"),
           "side": side,
           "contracts": delta_volume,
           "last": num(opt.get("last"), None),
           "bid": num(opt.get("bid"), None),
           "ask": num(opt.get("ask"), None),
       })

   classified = sum(totals[f"{kind}_{side}"] for kind in ("call", "put") for side in ("buy", "sell"))
   all_volume = classified + totals["call_unknown"] + totals["put_unknown"]
   score = totals["call_buy"] - 0.5 * totals["call_sell"] - totals["put_buy"] + 0.5 * totals["put_sell"]
   return {
       "events": sorted(events, key=lambda row: row["contracts"], reverse=True),
       "totals": totals,
       "directional_score": score,
       "classification_coverage_pct": classified / all_volume * 100 if all_volume else 0.0,
   }


def build_options_heatmap(
    chain: Iterable[Dict[str, Any]],
    live_events: Iterable[Dict[str, Any]],
    spot: float,
    level_window: Optional[float] = None,
) -> List[Dict[str, Any]]:
   """Combine live volume increments with current OI/GEX into strike cells."""
   by_strike = defaultdict(lambda: {
       "call_oi": 0, "put_oi": 0, "call_volume": 0, "put_volume": 0,
       "call_gex": 0.0, "put_gex": 0.0, "call_buy": 0, "call_sell": 0,
       "put_buy": 0, "put_sell": 0,
   })
   for opt in chain:
       if opt.get("strike") is None:
           continue
       strike = num(opt["strike"])
       if level_window is not None and abs(strike - spot) > level_window:
           continue
       row = by_strike[strike]
       kind = str(opt.get("option_type", "")).lower()
       if kind not in ("call", "put"):
           continue
       row[f"{kind}_oi"] += integer(opt.get("open_interest"))
       row[f"{kind}_volume"] += integer(opt.get("volume"))
       gamma = num((opt.get("greeks") or {}).get("gamma"))
       gex = gamma * integer(opt.get("open_interest")) * CONTRACT_MULTIPLIER * spot**2 * GEX_IV_SCALE
       row[f"{kind}_gex"] += gex if kind == "call" else -gex

   for event in live_events:
       row = by_strike.get(num(event.get("strike")))
       if row is None:
           continue
       side = str(event.get("side", "UNKNOWN")).lower()
       if side in ("buy", "sell"):
           row[f"{event['option_type']}_{side}"] += integer(event.get("contracts"))

   result = []
   for strike in sorted(by_strike):
       row = by_strike[strike]
       live_call = row["call_buy"] + row["call_sell"]
       live_put = row["put_buy"] + row["put_sell"]
       net_flow = row["call_buy"] - row["call_sell"] - row["put_buy"] + row["put_sell"]
       result.append({"strike": strike, **row, "live_volume": live_call + live_put,
                      "net_flow": net_flow, "distance": strike - spot})
   return result


def build_dealer_supply_demand(
    gex_table: Iterable[Dict[str, Any]],
    spot: float,
    level_window: Optional[float] = None,
) -> List[Dict[str, Any]]:
   """Return strike-level dealer supply and demand from estimated GEX.

   Positive call GEX is shown as supply; the magnitude of negative put GEX is
   shown as demand. These are positioning estimates, not observed dealer orders.
   """
   rows = []
   for row in gex_table:
       strike = num(row.get("strike"))
       if level_window is not None and abs(strike - spot) > level_window:
           continue
       supply = max(0.0, num(row.get("call_gex")))
       demand = max(0.0, -num(row.get("put_gex")))
       rows.append({
           "strike": strike,
           "supply": supply,
           "demand": demand,
           "net": supply - demand,
           "distance": strike - spot,
       })
   return rows




# ---------------------------------------------------------------------------
# GEX
# ---------------------------------------------------------------------------


def compute_gex(
   chain: Iterable[Dict[str, Any]],
   spot: float,
   contract_multiplier: int = CONTRACT_MULTIPLIER,
   gex_iv_scale: float = GEX_IV_SCALE,
   level_window: Optional[float] = None,
) -> Dict[str, Any]:
   """
   Estimated GEX using:


       call_gex = gamma * call_OI * 100 * spot^2 * 0.01
       put_gex  = -gamma * put_OI  * 100 * spot^2 * 0.01
   """
   spot = num(spot)
   if spot <= 0:
       raise ValueError("spot must be > 0")


   rows = defaultdict(
       lambda: {
           "call_oi": 0, "put_oi": 0,
           "call_volume": 0, "put_volume": 0,
           "call_gex": 0.0, "put_gex": 0.0,
       }
   )


   for opt in chain:
       if opt.get("strike") is None:
           continue


       strike = num(opt["strike"])
       typ = str(opt.get("option_type", "")).lower()
       oi = integer(opt.get("open_interest"))
       volume = integer(opt.get("volume"))


       if typ not in ("call", "put"):
           continue


       if typ == "call":
           rows[strike]["call_oi"] += oi
           rows[strike]["call_volume"] += volume
       else:
           rows[strike]["put_oi"] += oi
           rows[strike]["put_volume"] += volume


       gamma = (opt.get("greeks") or {}).get("gamma")
       if gamma is None or oi <= 0:
           continue


       gex = num(gamma) * oi * contract_multiplier * spot**2 * gex_iv_scale


       if typ == "call":
           rows[strike]["call_gex"] += gex
       else:
           rows[strike]["put_gex"] -= gex


   table = []
   for strike in sorted(rows):
       r = rows[strike]
       table.append({
           "strike": strike,
           "call_oi": r["call_oi"],
           "put_oi": r["put_oi"],
           "call_volume": r["call_volume"],
           "put_volume": r["put_volume"],
           "call_gex": r["call_gex"],
           "put_gex": r["put_gex"],
           "net_gex": r["call_gex"] + r["put_gex"],
       })


   if not table:
       return {
           "spot": spot, "net_gex": 0.0,
           "gross_call_gex": 0.0, "gross_put_gex": 0.0,
           "call_wall": None, "put_wall": None,
           "gamma_levels": compute_gamma_levels([], spot, max_distance=level_window),
           "call_gamma_wall": None, "put_gamma_wall": None,
           "call_wall_above": None, "put_wall_below": None,
           "upper_gamma_wall": None, "lower_gamma_wall": None,
           "local_gamma_transition": None, "flip_level": None, "table": [],
           "total_call_volume": 0, "total_put_volume": 0,
       }


   call_wall = max(table, key=lambda r: r["call_gex"])["strike"]
   put_wall = min(table, key=lambda r: r["put_gex"])["strike"]
   levels = compute_gamma_levels(table, spot, max_distance=level_window)
   transition = find_local_gamma_transition(table, spot)
   flip_level = (
   transition["interpolated"]
   if transition is not None
   else None)


   return {
   "spot": spot,
   "net_gex": sum(r["net_gex"] for r in table),


   "gross_call_gex": sum(r["call_gex"] for r in table),
   "gross_put_gex": sum(abs(r["put_gex"]) for r in table),


   # Chain-wide magnitude extremes. These are NOT resistance/support:
   # a call-gamma concentration can sit below spot and vice versa.
   "call_wall": call_wall,
   "put_wall": put_wall,


   # Spot-relative gamma structure, which is what the trade plan uses.
   "gamma_levels": levels,
   "call_gamma_wall": levels["call_gamma_wall"],
   "put_gamma_wall": levels["put_gamma_wall"],
   "call_wall_above": levels["call_wall_above"],
   "put_wall_below": levels["put_wall_below"],
   "upper_gamma_wall": levels["upper_gamma_wall"],
   "lower_gamma_wall": levels["lower_gamma_wall"],


   # New name
   "local_gamma_transition": transition,


   # Backward-compatible name
   "flip_level": flip_level,


   "table": table,


   "total_call_volume": sum(
       r["call_volume"] for r in table
   ),
   "total_put_volume": sum(
       r["put_volume"] for r in table
   ),
}




# ---------------------------------------------------------------------------
# Spot-relative gamma structure
# ---------------------------------------------------------------------------


def _gamma_level(row: Dict[str, Any], spot: float) -> Dict[str, Any]:
   call_gex = num(row.get("call_gex"))
   put_gex = num(row.get("put_gex"))
   return {
       "strike": row["strike"],
       "call_gex": call_gex,
       "put_gex": put_gex,
       "net_gex": num(row.get("net_gex")),
       "gamma_strength": abs(num(row.get("net_gex"))),
       "dominant_side": "call" if call_gex >= abs(put_gex) else "put",
       "distance": row["strike"] - spot,
       "side": "above" if row["strike"] > spot else "below",
   }




def _strongest(
   rows: List[Dict[str, Any]],
   key,
) -> Optional[Dict[str, Any]]:
   candidates = [r for r in rows if key(r) > 0]
   return max(candidates, key=key) if candidates else None




def compute_gamma_levels(
   table: List[Dict[str, Any]],
   spot: float,
   max_distance: Optional[float] = None,
   top_n: int = 3,
) -> Dict[str, Any]:
   """
   Locate gamma concentrations relative to spot.


   'Call wall' and 'put wall' describe which option type holds the gamma,
   not whether the strike acts as resistance or support, so the levels that
   matter intraday are the strongest concentrations above and below spot.


   `max_distance` restricts the search to strikes within N points of spot so
   that a far out-of-the-money strike cannot become the reference level.
   """
   empty = {
       "call_gamma_wall": None,
       "put_gamma_wall": None,
       "call_wall_above": None,
       "put_wall_below": None,
       "upper_gamma_wall": None,
       "lower_gamma_wall": None,
       "levels_above": [],
       "levels_below": [],
       "ladder": [],
   }
   if not table:
       return empty


   rows = [
       _gamma_level(r, spot)
       for r in table
       if max_distance is None or abs(r["strike"] - spot) <= max_distance
   ]
   if not rows:
       rows = [_gamma_level(r, spot) for r in table]


   above = [r for r in rows if r["side"] == "above"]
   below = [r for r in rows if r["side"] == "below"]


   call_gamma_wall = _strongest(rows, lambda r: r["call_gex"])
   put_gamma_wall = _strongest(rows, lambda r: -r["put_gex"])
   call_wall_above = _strongest(above, lambda r: r["call_gex"])
   put_wall_below = _strongest(below, lambda r: -r["put_gex"])
   upper = _strongest(above, lambda r: r["gamma_strength"])
   lower = _strongest(below, lambda r: r["gamma_strength"])


   by_strength = lambda rs: sorted(rs, key=lambda r: r["gamma_strength"], reverse=True)[:top_n]
   max_strength = max((r["gamma_strength"] for r in rows), default=0.0)


   ladder = [
       {**r, "strength_pct": r["gamma_strength"] / max_strength * 100 if max_strength else 0.0}
       for r in sorted(rows, key=lambda r: r["strike"], reverse=True)
   ]


   return {
       "call_gamma_wall": call_gamma_wall,
       "put_gamma_wall": put_gamma_wall,
       "call_wall_above": call_wall_above,
       "put_wall_below": put_wall_below,
       "upper_gamma_wall": upper,
       "lower_gamma_wall": lower,
       "levels_above": by_strength(above),
       "levels_below": by_strength(below),
       "ladder": ladder,
   }




def gamma_level_strike(level: Optional[Dict[str, Any]]) -> Optional[float]:
   return level["strike"] if level else None






def find_local_gamma_transition(
   table: List[Dict[str, Any]],
   spot: float,
) -> Optional[Dict[str, float]]:
   """
   Find the nearest strike-to-strike GEX sign transition around spot.


   This is intentionally preferred over a cumulative-from-lowest-strike
   'flip', because the cumulative result depends heavily on the selected
   chain range.
   """
   rows = sorted(table, key=lambda r: r["strike"])
   candidates = []


   for a, b in zip(rows, rows[1:]):
       ga, gb = num(a["net_gex"]), num(b["net_gex"])


       if ga == 0:
           level = a["strike"]
       elif ga * gb < 0:
           level = a["strike"] + (-ga / (gb - ga)) * (b["strike"] - a["strike"])
       else:
           continue


       candidates.append((
           abs(level - spot),
           {
               "lower": a["strike"],
               "upper": b["strike"],
               "interpolated": level,
           },
       ))


   return min(candidates, key=lambda x: x[0])[1] if candidates else None




def local_gex_rows(
   table: List[Dict[str, Any]],
   spot: float,
   count_each_side: int = 5,
) -> List[Dict[str, Any]]:
   rows = sorted(table, key=lambda r: abs(r["strike"] - spot))
   return sorted(
       rows[:2 * count_each_side + 1],
       key=lambda r: r["strike"],
   )




# ---------------------------------------------------------------------------
# Volume / leaderboards
# ---------------------------------------------------------------------------


def compute_volume_meter(gex: Dict[str, Any]) -> Dict[str, Any]:
   calls = integer(gex.get("total_call_volume"))
   puts = integer(gex.get("total_put_volume"))
   total = calls + puts


   return {
       "call_volume": calls,
       "put_volume": puts,
       "put_call_ratio": puts / calls if calls else None,
       "call_pct": calls / total * 100 if total else 50.0,
       "put_pct": puts / total * 100 if total else 50.0,
   }




def top_strikes(
   table: List[Dict[str, Any]],
   key: str,
   top_n: int = 5,
) -> List[Dict[str, Any]]:
   return sorted(
       [r for r in table if num(r.get(key)) != 0],
       key=lambda r: num(r[key]),
       reverse=True,
   )[:top_n]




def compute_leaderboards(
   table: List[Dict[str, Any]],
   top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
   return {
       "call_oi": top_strikes(table, "call_oi", top_n),
       "put_oi": top_strikes(table, "put_oi", top_n),
       "call_volume": top_strikes(table, "call_volume", top_n),
       "put_volume": top_strikes(table, "put_volume", top_n),
   }




# ---------------------------------------------------------------------------
# DEX
# ---------------------------------------------------------------------------


def compute_dex(
   chain: Iterable[Dict[str, Any]],
   spot: float,
   contract_multiplier: int = CONTRACT_MULTIPLIER,
) -> Dict[str, Any]:
   """Raw aggregate option delta represented by OI; not dealer DEX."""
   rows = defaultdict(lambda: {"call_dex": 0.0, "put_dex": 0.0})


   for opt in chain:
       if opt.get("strike") is None:
           continue


       typ = str(opt.get("option_type", "")).lower()
       oi = integer(opt.get("open_interest"))
       delta = (opt.get("greeks") or {}).get("delta")


       if typ not in ("call", "put") or oi <= 0 or delta is None:
           continue


       value = num(delta) * oi * contract_multiplier * spot
       rows[num(opt["strike"])]["call_dex" if typ == "call" else "put_dex"] += value


   table = []
   for strike in sorted(rows):
       r = rows[strike]
       table.append({
           "strike": strike,
           "call_dex": r["call_dex"],
           "put_dex": r["put_dex"],
           "net_dex": r["call_dex"] + r["put_dex"],
       })


   return {
       "spot": spot,
       "net_dex": sum(r["net_dex"] for r in table),
       "table": table,
   }




# ---------------------------------------------------------------------------
# VEX
# ---------------------------------------------------------------------------


def compute_vex(
   chain: Iterable[Dict[str, Any]],
   contract_multiplier: int = CONTRACT_MULTIPLIER,
) -> Dict[str, Any]:
   """Gross call/put vega exposure. Do not interpret as signed dealer vega."""
   rows = defaultdict(lambda: {"call_vex": 0.0, "put_vex": 0.0})


   for opt in chain:
       if opt.get("strike") is None:
           continue


       typ = str(opt.get("option_type", "")).lower()
       oi = integer(opt.get("open_interest"))
       vega = (opt.get("greeks") or {}).get("vega")


       if typ not in ("call", "put") or oi <= 0 or vega is None:
           continue


       value = num(vega) * oi * contract_multiplier
       rows[num(opt["strike"])]["call_vex" if typ == "call" else "put_vex"] += value


   table = [
       {"strike": strike, **rows[strike]}
       for strike in sorted(rows)
   ]


   return {
       "total_call_vega": sum(r["call_vex"] for r in table),
       "total_put_vega": sum(r["put_vex"] for r in table),
       "table": table,
   }




# ---------------------------------------------------------------------------
# Max pain
# ---------------------------------------------------------------------------


def compute_max_pain(chain: Iterable[Dict[str, Any]]) -> Optional[float]:
   strikes = sorted({
       num(o["strike"])
       for o in chain
       if o.get("strike") is not None
   })
   if not strikes:
       return None


   calls = defaultdict(int)
   puts = defaultdict(int)


   for o in chain:
       if o.get("strike") is None:
           continue
       strike = num(o["strike"])
       oi = integer(o.get("open_interest"))
       typ = str(o.get("option_type", "")).lower()


       if typ == "call":
           calls[strike] += oi
       elif typ == "put":
           puts[strike] += oi


   best_strike, best_payout = None, None


   for test in strikes:
       payout = sum(
           (test - strike) * oi
           for strike, oi in calls.items()
           if test > strike
       )
       payout += sum(
           (strike - test) * oi
           for strike, oi in puts.items()
           if test < strike
       )


       if best_payout is None or payout < best_payout:
           best_payout, best_strike = payout, test


   return best_strike




# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


def assess_data_quality(
   chain: Iterable[Dict[str, Any]],
   gex: Dict[str, Any],
) -> Dict[str, Any]:
   chain = list(chain)
   n = len(chain)


   missing_gamma = sum(
       1 for o in chain if (o.get("greeks") or {}).get("gamma") is None
   )
   missing_delta = sum(
       1 for o in chain if (o.get("greeks") or {}).get("delta") is None
   )
   missing_vega = sum(
       1 for o in chain if (o.get("greeks") or {}).get("vega") is None
   )
   zero_oi = sum(1 for o in chain if integer(o.get("open_interest")) <= 0)


   invalid_quotes = 0
   for o in chain:
       if o.get("bid") is None or o.get("ask") is None:
           continue
       bid, ask = num(o["bid"]), num(o["ask"])
       if bid < 0 or ask < bid:
           invalid_quotes += 1


   warnings = []
   if n and missing_gamma / n > 0.10:
       warnings.append("More than 10% of contracts are missing gamma.")
   if n and missing_delta / n > 0.10:
       warnings.append("More than 10% of contracts are missing delta.")
   if n and missing_vega / n > 0.10:
       warnings.append("More than 10% of contracts are missing vega.")
   if invalid_quotes:
       warnings.append(f"{invalid_quotes} contracts have invalid/crossed quotes.")
   if zero_oi:
       warnings.append(f"{zero_oi} contracts have zero/empty OI.")


   if not gex["table"]:
       warnings.append("No valid strikes available for GEX.")


   return {
       "contracts": n,
       "missing_gamma": missing_gamma,
       "missing_delta": missing_delta,
       "missing_vega": missing_vega,
       "invalid_quotes": invalid_quotes,
       "zero_oi": zero_oi,
       "warnings": warnings,
       "ok": not warnings,
   }




# ---------------------------------------------------------------------------
# Trade plan
# ---------------------------------------------------------------------------


def build_trade_plan(
   spot: float,
   net_gex: float,
   gamma_levels: Dict[str, Any],
   transition: Optional[Dict[str, float]],
   flow: Dict[str, Any],
   price_levels: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
   """
   Conditional intraday plan. It does not predict price.


   Triggers are derived from gamma concentrations above/below spot and the
   gamma transition (the regime boundary), not from call-wall/put-wall labels.
   """
   regime = (
       "POSITIVE_GAMMA" if net_gex > 0
       else "NEGATIVE_GAMMA" if net_gex < 0
       else "NEUTRAL_GAMMA"
   )


   transition_level = transition["interpolated"] if transition else None
   upper = gamma_level_strike(gamma_levels.get("upper_gamma_wall"))
   lower = gamma_level_strike(gamma_levels.get("lower_gamma_wall"))


   # Above the transition the regime itself changes, so it outranks any wall.
   bullish_trigger = (
       transition_level
       if transition_level is not None and transition_level > spot
       else upper
   )
   bearish_trigger = lower


   upside = [
       level["strike"]
       for level in sorted(
           gamma_levels.get("levels_above", []),
           key=lambda r: r["strike"],
       )
   ]
   downside = [
       level["strike"]
       for level in sorted(
           gamma_levels.get("levels_below", []),
           key=lambda r: -r["strike"],
       )
   ]


   price_context = None
   if price_levels:
       price_context = nearest_price_levels(price_levels.get("levels", []), spot)


   return {
       "regime": regime,
       "gamma_transition": transition_level,
       "bullish_trigger": bullish_trigger,
       "bearish_trigger": bearish_trigger,
       "bullish_targets": upside,
       "bearish_targets": downside,
       "gamma_above": gamma_levels.get("levels_above", []),
       "gamma_below": gamma_levels.get("levels_below", []),
       "price_levels_above": (price_context or {}).get("above", []),
       "price_levels_below": (price_context or {}).get("below", []),
       "long_rule": (
           "Long only after reclaim/hold of the bullish trigger with "
           "supporting aggressor flow; a reclaim of the gamma transition is "
           "the stronger signal because it changes the regime."
       ),
       "short_rule": (
           "Short only after a break and failed retest of the bearish "
           "trigger with supporting bearish flow."
       ),
       "negative_gamma_behavior": (
           "Negative GEX: favor momentum confirmation; avoid blindly fading "
           "breaks because moves can accelerate."
       ),
       "positive_gamma_behavior": (
           "Positive GEX: expect more mean-reversion/stabilization unless "
           "a strong catalyst overwhelms the positioning."
       ),
       "flow_score": flow.get("directional_score", 0.0),
   }


def build_trade_idea(snapshot: Dict[str, Any]) -> Dict[str, Any]:
   """Create a conditional BUY/SELL idea from SMC and GEX confluence."""
   spot = num(snapshot.get("spot"))
   gamma_levels = snapshot.get("gamma_levels") or {}
   smc = snapshot.get("smc") or {}
   multi_timeframe = snapshot.get("multi_timeframe") or {}
   htf_bias = multi_timeframe.get("htf_bias", "NEUTRAL")
   timeframes = multi_timeframe.get("timeframes") or {}
   timeframe_biases = {
       timeframe: (timeframes.get(timeframe) or {}).get("bias", "NEUTRAL")
       for timeframe in ("15M", "1H", "4H")
   }
   entry_1m = multi_timeframe.get("entry_1m") or {}
   price_levels = snapshot.get("price_levels") or []
   net_gex = num(snapshot.get("net_gex"))
   net_dex = num(snapshot.get("net_dex"))
   flow_score = num(snapshot.get("flow_score"))
   flip = num(snapshot.get("flip_level"), math.nan)

   bullish_score = 0
   bearish_score = 0
   reasons = []
   if smc.get("bias") == "BULLISH":
       bullish_score += 1
       reasons.append("higher-high/higher-low structure")
   elif smc.get("bias") == "BEARISH":
       bearish_score += 1
       reasons.append("lower-high/lower-low structure")
   if smc.get("bos") == "BULLISH" or smc.get("liquidity_sweep") == "BULLISH":
       bullish_score += 2
       reasons.append("bullish BOS/liquidity sweep")
   elif smc.get("bos") == "BEARISH" or smc.get("liquidity_sweep") == "BEARISH":
       bearish_score += 2
       reasons.append("bearish BOS/liquidity sweep")
   active_fvg = smc.get("active_fvg") or {}
   if active_fvg.get("direction") == "BULLISH":
       bullish_score += 1
       reasons.append("bullish FVG support")
   elif active_fvg.get("direction") == "BEARISH":
       bearish_score += 1
       reasons.append("bearish FVG resistance")
   inverse_fvg = (smc.get("inverse_fvgs") or [None])[-1] or {}
   if inverse_fvg.get("direction") == "BULLISH":
       bullish_score += 1
       reasons.append("bullish inverse FVG")
   elif inverse_fvg.get("direction") == "BEARISH":
       bearish_score += 1
       reasons.append("bearish inverse FVG")
   if net_gex >= 0:
       bullish_score += 1
   else:
       bearish_score += 1
   if math.isfinite(flip):
       if spot >= flip:
           bullish_score += 1
           reasons.append("above gamma flip")
       else:
           bearish_score += 1
           reasons.append("below gamma flip")
   if net_dex > 0 or flow_score > 0:
       bullish_score += 1
   elif net_dex < 0 or flow_score < 0:
       bearish_score += 1

   bias_15m = timeframe_biases["15M"]
   bias_1h = timeframe_biases["1H"]
   bias_4h = timeframe_biases["4H"]
   if bias_1h == bias_4h and bias_1h in {"BULLISH", "BEARISH"}:
       candidate_direction = "BUY" if bias_1h == "BULLISH" else "SELL"
       entry_bias = "BULLISH" if candidate_direction == "BUY" else "BEARISH"
       if bias_15m == entry_bias:
           direction = candidate_direction
           alignment_reason = "15M/1H/4H aligned"
       else:
           direction = "WAIT"
           alignment_reason = "15M does not confirm 1H/4H"
   else:
       direction = "WAIT"
       alignment_reason = "1H and 4H disagree or are neutral"
   if direction != "WAIT":
       reasons.insert(0, alignment_reason)
   entry_direction = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(direction)
   entry_confirmation = (
       entry_direction is not None and (
           entry_1m.get("bias") == entry_direction
           or entry_1m.get("bos") == entry_direction
           or entry_1m.get("liquidity_sweep") == entry_direction
       )
   )
   entry_confirmation_pending = direction != "WAIT" and not entry_confirmation
   if entry_confirmation_pending:
       direction = "WAIT"
   entry_zone = (
       entry_1m.get("active_fvg")
       or entry_1m.get("order_block")
       or entry_1m.get("rejection_block")
   )
   entry_stddev = entry_1m.get("stddev") or {}
   if direction == "WAIT":
       reasons = ["1M entry confirmation pending"] if entry_confirmation_pending else [alignment_reason]
   elif entry_confirmation:
       reasons.append("1M entry confirmation")
   else:
       reasons.append("1M entry confirmation pending")
   above = sorted(
       [num(x.get("price")) for x in price_levels if num(x.get("price")) > spot]
       + [num(x.get("strike")) for x in gamma_levels.get("levels_above", [])],
   )
   below = sorted(
       [num(x.get("price")) for x in price_levels if num(x.get("price")) < spot]
       + [num(x.get("strike")) for x in gamma_levels.get("levels_below", [])],
       reverse=True,
   )
   above.extend(
       num(level) for level in (entry_stddev.get("upper_1"), entry_stddev.get("upper_2"))
       if level is not None and level > spot
   )
   below.extend(
       num(level) for level in (entry_stddev.get("lower_1"), entry_stddev.get("lower_2"))
       if level is not None and level < spot
   )
   external = smc.get("external_liquidity") or {}
   external_target = None
   if direction == "BUY":
       external_target = (external.get("buy_side") or {}).get("price")
   elif direction == "SELL":
       external_target = (external.get("sell_side") or {}).get("price")
   if external_target is not None:
       if direction == "BUY" and external_target > spot:
           above.append(num(external_target))
       elif direction == "SELL" and external_target < spot:
           below.append(num(external_target))
   above = sorted(set(above))
   below = sorted(set(below), reverse=True)
   order_block = smc.get("order_block") or {}
   tp_levels = []
   stop = None
   risk_points = None
   if direction == "BUY":
       stop_candidates = [
           x for x in below + [
               num(order_block.get("low"), math.nan),
               num(entry_zone.get("low"), math.nan) if entry_zone else math.nan,
           ] if math.isfinite(x) and x < spot
       ]
       stop = max(stop_candidates) if stop_candidates else spot * 0.995
       target_levels = above
       direction_sign = 1
   elif direction == "SELL":
       stop_candidates = [
           x for x in above + [
               num(order_block.get("high"), math.nan),
               num(entry_zone.get("high"), math.nan) if entry_zone else math.nan,
           ] if math.isfinite(x) and x > spot
       ]
       stop = min(stop_candidates) if stop_candidates else spot * 1.005
       target_levels = below
       direction_sign = -1
   if direction != "WAIT":
       risk_points = abs(spot - stop)
       target_levels = sorted(set(target_levels), reverse=direction_sign < 0)
       for multiple in (1, 2, 3):
           minimum_distance = risk_points * multiple
           if tp_levels:
               minimum_distance = max(
                   minimum_distance,
                   abs(tp_levels[-1] - spot) + risk_points,
               )
           target = next((
               level for level in target_levels
               if direction_sign * (level - spot) >= minimum_distance
           ), None)
           if target is None:
               target = spot + direction_sign * minimum_distance
           tp_levels.append(round(target, 2))

   return {
       "action": direction,
       "entry": spot,
       "target": tp_levels[0] if tp_levels else None,
       "tp1": tp_levels[0] if tp_levels else None,
       "tp2": tp_levels[1] if tp_levels else None,
       "tp3": tp_levels[2] if tp_levels else None,
       "stop_loss": round(stop, 2) if stop is not None else None,
       "risk_points": round(risk_points, 2) if risk_points is not None else None,
       "reward_points": round(abs(tp_levels[0] - spot), 2) if tp_levels else None,
       "smc_bias": smc.get("bias", "NEUTRAL"),
    "trend_strength": smc.get("trend_strength", "NEUTRAL"),
    "trend_strength_score": smc.get("trend_strength_score", 0),
    "strength_signals": smc.get("strength_signals", []),
    "weakness_signals": smc.get("weakness_signals", []),
    "strength_level": smc.get("strength_level"),
    "strength_condition": smc.get("strength_condition"),
    "weakness_level": smc.get("weakness_level"),
    "weakness_condition": smc.get("weakness_condition"),
    "amd_phase": smc.get("amd_phase", "UNCONFIRMED"),
    "amd_direction": smc.get("amd_direction"),
    "amd_evidence": smc.get("amd_evidence"),
       "bos": smc.get("bos"),
       "liquidity_sweep": smc.get("liquidity_sweep"),
    "active_fvg": active_fvg,
    "inverse_fvg": inverse_fvg,
    "internal_liquidity": smc.get("internal_liquidity", []),
    "external_liquidity": external,
    "liquidity_path": smc.get("liquidity_path"),
    "htf_bias": htf_bias,
    "timeframe_biases": timeframe_biases,
    "timeframe_alignment": "ALIGNED" if bias_1h == bias_4h == bias_15m and bias_1h in {"BULLISH", "BEARISH"} else alignment_reason,
    "htf_bullish_count": multi_timeframe.get("htf_bullish_count", 0),
    "htf_bearish_count": multi_timeframe.get("htf_bearish_count", 0),
    "entry_1m_bias": entry_1m.get("bias", "NEUTRAL"),
    "entry_1m_confirmation": "CONFIRMED" if entry_confirmation else "PENDING",
    "entry_1m_zone": entry_zone,
    "entry_1m_stddev": entry_1m.get("stddev", {}),
    "confidence": min(95, 60 + abs(bullish_score - bearish_score) * 5) if direction != "WAIT" else 0,
    "reason": ", ".join(reasons[:4]) or alignment_reason,
       "risk_note": "Conditional heuristic; wait for price confirmation and use defined risk.",
   }




def nearest_price_levels(
   levels: List[Dict[str, Any]],
   spot: float,
   count: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
   """Closest PDH/PDL/session/VWAP style levels above and below spot."""
   above = sorted([x for x in levels if num(x.get("price")) > spot], key=lambda x: x["price"])
   below = sorted([x for x in levels if num(x.get("price")) < spot], key=lambda x: -x["price"])
   return {"above": above[:count], "below": below[:count]}






# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def analyze_chain(
   chain: Iterable[Dict[str, Any]],
   spot: float,
   top_n: int = 5,
   flow_tolerance: float = 0.01,
   level_window: Optional[float] = None,
   price_levels: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
   """
   Full analysis in one call.


   Expected option fields:
     strike, option_type, open_interest, volume,
     bid, ask, last,
     greeks.gamma, greeks.delta, greeks.vega


   `price_levels` is the optional output of price_levels.build_price_levels.
   """
   chain = list(chain)


   gex = compute_gex(chain, spot, level_window=level_window)
   flow = compute_flow(chain, flow_tolerance)
   dex = compute_dex(chain, spot)
   vex = compute_vex(chain)
   volume = compute_volume_meter(gex)
   leaders = compute_leaderboards(gex["table"], top_n)
   max_pain = compute_max_pain(chain)
   local = local_gex_rows(gex["table"], spot)


   plan = build_trade_plan(
       spot=spot,
       net_gex=gex["net_gex"],
       gamma_levels=gex["gamma_levels"],
       transition=gex["local_gamma_transition"],
       flow=flow,
       price_levels=price_levels,
   )


   quality = assess_data_quality(chain, gex)


   return {
       "spot": spot,
       "gex": gex,
       "flow": flow,
       "dex": dex,
       "vex": vex,
       "volume": volume,
       "leaderboards": leaders,
       "max_pain": max_pain,
       "local_gex": local,
       "gamma_levels": gex["gamma_levels"],
       "price_levels": price_levels,
       "trade_plan": plan,
       "data_quality": quality,
   }




# ---------------------------------------------------------------------------
# Live snapshot monitoring
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
   timestamp: float
   spot: float
   net_gex: float
   upper_gamma_wall: Optional[float]
   lower_gamma_wall: Optional[float]
   call_wall_above: Optional[float]
   put_wall_below: Optional[float]
   local_transition: Optional[float]
   net_dex: float
   call_vega: float
   put_vega: float
   call_volume: int
   put_volume: int
   flow_score: float




class SnapshotMonitor:
   """
   Keep recent summaries and detect suspicious GEX jumps.


   For 0DTE, require several consecutive snapshots before acting on a
   perceived GEX regime change.
   """


   def __init__(
       self,
       maxlen: int = 120,
       gex_jump_b: float = 3.0,
       confirmation_snapshots: int = 3,
   ):
       self.history: Deque[Snapshot] = deque(maxlen=maxlen)
       self.gex_jump = gex_jump_b * 1e9
       self.confirmation_snapshots = confirmation_snapshots


   def add(self, result: Dict[str, Any]) -> Dict[str, Any]:
       g = result["gex"]
       d = result["dex"]
       v = result["vex"]
       vol = result["volume"]
       f = result["flow"]


       transition = g["local_gamma_transition"]
       levels = g["gamma_levels"]
       snapshot = Snapshot(
           timestamp=time.time(),
           spot=result["spot"],
           net_gex=g["net_gex"],
           upper_gamma_wall=gamma_level_strike(levels["upper_gamma_wall"]),
           lower_gamma_wall=gamma_level_strike(levels["lower_gamma_wall"]),
           call_wall_above=gamma_level_strike(levels["call_wall_above"]),
           put_wall_below=gamma_level_strike(levels["put_wall_below"]),
           local_transition=(
               transition["interpolated"] if transition else None
           ),
           net_dex=d["net_dex"],
           call_vega=v["total_call_vega"],
           put_vega=v["total_put_vega"],
           call_volume=vol["call_volume"],
           put_volume=vol["put_volume"],
           flow_score=f["directional_score"],
       )


       previous = self.history[-1] if self.history else None
       events = []


       if previous:
           dgex = snapshot.net_gex - previous.net_gex
           dt = max(0.001, snapshot.timestamp - previous.timestamp)


           if abs(dgex) >= self.gex_jump and dt <= 2.0:
               events.append(
                   f"WARNING: GEX moved {dgex / 1e9:+.2f}B in {dt:.2f}s; "
                   "verify quote/Greek freshness."
               )


           if (previous.net_gex >= 0) != (snapshot.net_gex >= 0):
               events.append(
                   f"NET GEX REGIME FLIP: "
                   f"{previous.net_gex / 1e9:+.2f}B -> "
                   f"{snapshot.net_gex / 1e9:+.2f}B"
               )


           for name, old, new in (
               ("UPPER GAMMA WALL", previous.upper_gamma_wall, snapshot.upper_gamma_wall),
               ("LOWER GAMMA WALL", previous.lower_gamma_wall, snapshot.lower_gamma_wall),
               ("CALL WALL ABOVE", previous.call_wall_above, snapshot.call_wall_above),
               ("PUT WALL BELOW", previous.put_wall_below, snapshot.put_wall_below),
           ):
               if old != new:
                   events.append(f"{name} MOVED: {old} -> {new}")
               elif new is not None and (previous.spot - new) * (snapshot.spot - new) < 0:
                   events.append(f"SPOT CROSSED {name} @ {new}")


       self.history.append(snapshot)


       return {
           "snapshot": asdict(snapshot),
           "events": events,
           "gex_change_confirmed": self.gex_change_confirmed(),
       }


   def gex_change_confirmed(self) -> bool:
       if len(self.history) < self.confirmation_snapshots:
           return False


       recent = list(self.history)[-self.confirmation_snapshots:]
       signs = [x.net_gex >= 0 for x in recent]
       return all(x == signs[0] for x in signs)


   def recent(self, n: int = 10) -> List[Dict[str, Any]]:
       return [asdict(x) for x in list(self.history)[-n:]]




# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------


def format_analysis(result: Dict[str, Any]) -> str:
   g = result["gex"]
   f = result["flow"]
   d = result["dex"]
   v = result["vex"]
   vm = result["volume"]
   p = result["trade_plan"]
   q = result["data_quality"]


   t = g["local_gamma_transition"]
   transition = (
       f"{t['interpolated']:.1f} ({t['lower']}-{t['upper']})"
       if t else "None"
   )


   def level_text(level):
       if not level:
           return "None"
       return (
           f"{level['strike']:.0f} "
           f"({level['dominant_side']} gamma, {level['net_gex'] / 1e9:+.2f}B)"
       )


   lv = g["gamma_levels"]


   lines = [
       "=" * 72,
       "SPX OPTIONS POSITIONING",
       "=" * 72,
       f"SPOT                 {result['spot']:.2f}",
       f"NET GEX              {g['net_gex'] / 1e9:+.2f}B",
       f"LOCAL GEX TRANSITION {transition}",
       "",
       "GAMMA STRUCTURE (relative to spot)",
       f"STRONGEST ABOVE      {level_text(lv['upper_gamma_wall'])}",
       f"CALL WALL ABOVE      {level_text(lv['call_wall_above'])}",
       f"PUT WALL BELOW       {level_text(lv['put_wall_below'])}",
       f"STRONGEST BELOW      {level_text(lv['lower_gamma_wall'])}",
       "",
       f"RAW NET DEX          {d['net_dex'] / 1e9:+.2f}B",
       f"GROSS CALL VEGA      {v['total_call_vega'] / 1e6:+.2f}M",
       f"GROSS PUT VEGA       {v['total_put_vega'] / 1e6:+.2f}M",
       "",
       f"CALL VOLUME          {vm['call_volume']:,}",
       f"PUT VOLUME           {vm['put_volume']:,}",
       f"PUT/CALL             {vm['put_call_ratio']:.2f}" if vm["put_call_ratio"] is not None else "PUT/CALL             N/A",
       "",
       "ESTIMATED AGGRESSOR FLOW",
       f"CALL BUY             {f['totals']['call_buy']:,}",
       f"CALL SELL            {f['totals']['call_sell']:,}",
       f"PUT BUY              {f['totals']['put_buy']:,}",
       f"PUT SELL             {f['totals']['put_sell']:,}",
       f"FLOW SCORE           {f['directional_score']:+,.0f}",
       f"FLOW COVERAGE        {f['classification_coverage_pct']:.1f}%",
       "",
       "CONDITIONAL TRADE PLAN",
       f"LONG TRIGGER         {p['bullish_trigger']}",
       f"SHORT TRIGGER        {p['bearish_trigger']}",
       f"UPSIDE TARGETS       {p['bullish_targets']}",
       f"DOWNSIDE TARGETS     {p['bearish_targets']}",
       "",
       "DATA QUALITY",
       f"OK                   {q['ok']}",
   ]


   if q["warnings"]:
       lines.append("WARNINGS:")
       lines.extend(f"  - {w}" for w in q["warnings"])


   return "\n".join(lines)


def detect_events(prev, curr):
   """Detect important changes between two GEX snapshots."""


   events = []


   if prev is None or curr is None:
       return events


   # Net GEX regime flip
   prev_gex = prev.get("net_gex")
   curr_gex = curr.get("net_gex")


   if prev_gex is not None and curr_gex is not None:
       if (prev_gex >= 0) != (curr_gex >= 0):
           direction = "POSITIVE" if curr_gex >= 0 else "NEGATIVE"


           events.append(
               f"Net GEX regime flipped to {direction} "
               f"({curr_gex / 1e9:.2f}B)"
           )


   # Spot-relative gamma structure movement
   for key, label in (
       ("upper_gamma_wall", "Upper gamma wall"),
       ("lower_gamma_wall", "Lower gamma wall"),
       ("call_wall_above", "Call wall above spot"),
       ("put_wall_below", "Put wall below spot"),
   ):
       prev_level = prev.get(key)
       curr_level = curr.get(key)


       if prev_level != curr_level:
           events.append(f"{label} moved {prev_level} -> {curr_level}")


   # Gamma flip movement
   prev_flip = prev.get("flip_level")
   curr_flip = curr.get("flip_level")


   if (
       prev_flip is not None
       and curr_flip is not None
       and abs(prev_flip - curr_flip) > 1
   ):
       events.append(
           f"Gamma flip moved "
           f"{prev_flip:.1f} -> {curr_flip:.1f}"
       )


   # Spot crossing gamma flip
   prev_spot = prev.get("spot")
   curr_spot = curr.get("spot")


   if (
       prev_spot is not None
       and curr_spot is not None
       and prev_flip is not None
       and curr_flip is not None
   ):
       was_above = prev_spot >= prev_flip
       now_above = curr_spot >= curr_flip


       if was_above != now_above:
           direction = "above" if now_above else "below"


           events.append(
               f"Spot crossed gamma flip, now {direction} "
               f"({curr_spot:.2f} vs {curr_flip:.1f})"
           )


   # Spot crossing gamma concentrations
   if prev_spot is not None and curr_spot is not None:


       for key, label in (
           ("upper_gamma_wall", "upper gamma wall"),
           ("lower_gamma_wall", "lower gamma wall"),
       ):
           level = curr.get(key)


           if level is None:
               continue


           crossed = (prev_spot - level) * (curr_spot - level) < 0


           if crossed:
               events.append(f"Spot crossed {label} at {level}")


   # Price-structure levels (PDH/PDL/session highs and lows)
   if prev_spot is not None and curr_spot is not None:
       for level in curr.get("price_levels") or []:
           price = level.get("price")


           if price is None:
               continue


           if (prev_spot - price) * (curr_spot - price) < 0:
               direction = "above" if curr_spot > price else "below"
               events.append(
                   f"Spot crossed {level.get('name')} "
                   f"{price:.2f}, now {direction}"
               )


   return events


def generate_trade_plan_events(gex):
   """
   Generate actionable trade-plan events from the current snapshot.


   This is a STRUCTURAL signal generator, not an automatic trade executor.
   Levels are read relative to spot: the gamma transition defines the regime,
   and the concentrations above/below spot define the intraday decision points.
   """


   spot = gex.get("spot")
   net_gex = gex.get("net_gex", 0.0)
   upper = gex.get("upper_gamma_wall")
   lower = gex.get("lower_gamma_wall")
   call_above = gex.get("call_wall_above")
   put_below = gex.get("put_wall_below")
   flip = gex.get("flip_level")
   net_dex = gex.get("net_dex", 0.0)
   price_levels = gex.get("price_levels") or []


   events = []


   if spot is None:
       return events


   # ---------------------------------------------------------
   # Basic regime
   # ---------------------------------------------------------


   if net_gex < -10e9:
       regime = "VERY NEGATIVE GAMMA"
       regime_level = "critical"
   elif net_gex < 0:
       regime = "NEGATIVE GAMMA"
       regime_level = "bearish"
   elif net_gex > 10e9:
       regime = "STRONG POSITIVE GAMMA"
       regime_level = "bullish"
   else:
       regime = "POSITIVE GAMMA"
       regime_level = "neutral"


   events.append({
       "type": "REGIME",
       "severity": regime_level,
       "title": regime,
       "message": (
           f"Net GEX {net_gex / 1e9:.2f}B"
       ),
   })


   # ---------------------------------------------------------
   # Gamma transition: the regime boundary, the most important level
   # ---------------------------------------------------------


   if flip is not None:
       distance = spot - flip


       if distance < 0:
           events.append({
               "type": "GAMMA_FLIP",
               "severity": "bearish",
               "title": "BELOW GAMMA FLIP",
               "message": (
                   f"SPX {abs(distance):.1f} pts below gamma flip {flip:.1f}; "
                   f"regime only changes on a reclaim."
               ),
               "trigger": flip,
               "direction": "LONG",
           })


       elif distance > 0:
           events.append({
               "type": "GAMMA_FLIP",
               "severity": "bullish",
               "title": "ABOVE GAMMA FLIP",
               "message": (
                   f"SPX {distance:.1f} pts above "
                   f"gamma flip {flip:.1f}"
               ),
               "trigger": flip,
               "direction": "LONG",
           })


   # ---------------------------------------------------------
   # Gamma structure around spot
   # ---------------------------------------------------------


   if upper is not None:
       events.append({
           "type": "GAMMA_ABOVE",
           "severity": "neutral",
           "title": "GAMMA CONCENTRATION ABOVE",
           "message": (
               f"Strongest gamma above spot at {upper:.0f} "
               f"({upper - spot:+.1f} pts)."
           ),
           "trigger": upper,
       })


   if lower is not None:
       events.append({
           "type": "GAMMA_BELOW",
           "severity": "neutral",
           "title": "GAMMA CONCENTRATION BELOW",
           "message": (
               f"Strongest gamma below spot at {lower:.0f} "
               f"({lower - spot:+.1f} pts)."
           ),
           "trigger": lower,
       })


   if upper is not None and lower is not None and lower <= spot <= upper:
       events.append({
           "type": "ZONE",
           "severity": "warning",
           "title": "GAMMA DECISION ZONE",
           "message": f"{lower:.0f}–{upper:.0f} around current price",
       })


   # ---------------------------------------------------------
   # Short / long structure
   # ---------------------------------------------------------


   if lower is not None:
       if spot < lower:
           events.append({
               "type": "SHORT",
               "severity": "critical",
               "title": "SHORT TRIGGER ACTIVE",
               "message": (
                   f"SPX below gamma support {lower:.0f}. "
                   f"Look for a failed retest from below."
               ),
               "entry": lower,
               "trigger": lower,
               "direction": "SHORT",
           })
       elif spot <= lower + 2:
           events.append({
               "type": "SHORT_SETUP",
               "severity": "warning",
               "title": "SHORT SETUP",
               "message": (
                   f"Watch {lower:.0f}. "
                   f"Break + failed retest confirms short."
               ),
               "trigger": lower,
               "direction": "SHORT",
           })


   if upper is not None and spot >= upper:
       events.append({
           "type": "RECLAIM",
           "severity": "bullish",
           "title": "ABOVE UPPER GAMMA WALL",
           "message": (
               f"SPX accepted above {upper:.0f}. "
               f"Needs to hold to stay constructive."
           ),
           "trigger": upper,
           "direction": "LONG",
       })


   if call_above is not None and put_below is not None:
       events.append({
           "type": "STRUCTURE",
           "severity": "neutral",
           "title": "GAMMA WALLS",
           "message": (
               f"Call gamma above {call_above:.0f} / "
               f"put gamma below {put_below:.0f}."
           ),
       })


   # ---------------------------------------------------------
   # Price-structure confluence
   # ---------------------------------------------------------


   nearest = nearest_price_levels(price_levels, spot)


   for level in nearest["below"][:1]:
       events.append({
           "type": "PRICE_LEVEL_BELOW",
           "severity": "neutral",
           "title": f"SUPPORT REFERENCE: {level['name']}",
           "message": (
               f"{level['name']} at {level['price']:.2f} "
               f"({level['price'] - spot:+.1f} pts)."
           ),
           "trigger": level["price"],
       })


   for level in nearest["above"][:1]:
       events.append({
           "type": "PRICE_LEVEL_ABOVE",
           "severity": "neutral",
           "title": f"RESISTANCE REFERENCE: {level['name']}",
           "message": (
               f"{level['name']} at {level['price']:.2f} "
               f"({level['price'] - spot:+.1f} pts)."
           ),
           "trigger": level["price"],
       })


   if net_gex < 0 and lower is not None and spot < lower:
       broken = [x["name"] for x in price_levels if x["price"] > spot and x["kind"] != "vwap"]
       if broken:
           events.append({
               "type": "CONFLUENCE",
               "severity": "critical",
               "title": "SHORT CONFIRMATION CONFLUENCE",
               "message": (
                   f"Below {lower:.0f} gamma support and below "
                   f"{', '.join(broken[:3])} with negative gamma."
               ),
               "direction": "SHORT",
               "trigger": lower,
           })


   # ---------------------------------------------------------
   # DEX confirmation
   # ---------------------------------------------------------


   if net_dex < -1e9:


       events.append({
           "type": "DEX",
           "severity": "bearish",
           "title": "NEGATIVE DEX",
           "message": (
               f"Aggregate DEX {net_dex / 1e9:.2f}B"
           ),
       })


   elif net_dex > 1e9:


       events.append({
           "type": "DEX",
           "severity": "bullish",
           "title": "POSITIVE DEX",
           "message": (
               f"Aggregate DEX {net_dex / 1e9:.2f}B"
           ),
       })


   return events
# ============================================================
# PRICE / GEX MARKET COMMENTARY
# ============================================================


def _fmt_price(price):
   if price is None:
       return "N/A"
   return f"{price:,.2f}"




def _fmt_b(value):
   if value is None:
       return "N/A"
   return f"{value / 1e9:.2f}B"




def _nearest_level(levels, spot, direction):
   """
   Find nearest price level above/below spot.


   direction:
       "above" -> smallest level > spot
       "below" -> largest level < spot
   """
   valid = [
       x for x in levels
       if x.get("price") is not None
   ]


   if direction == "above":
       candidates = [x for x in valid if x["price"] > spot]
       return min(candidates, key=lambda x: x["price"], default=None)


   candidates = [x for x in valid if x["price"] < spot]
   return max(candidates, key=lambda x: x["price"], default=None)




def _nearest_gex_row(gex_result, spot, direction):
   rows = gex_result.get("table", [])


   if direction == "above":
       rows = [r for r in rows if r["strike"] > spot]
       return min(rows, key=lambda r: r["strike"], default=None)


   rows = [r for r in rows if r["strike"] < spot]
   return max(rows, key=lambda r: r["strike"], default=None)




def get_gex_context(gex_result, spot, radius=50):
   """
   Extract the most important GEX information around spot.


   Returns:
       regime
       strongest_positive
       strongest_negative
       nearest_positive_above
       nearest_negative_below
   """


   table = gex_result.get("table", [])


   nearby = [
       r for r in table
       if abs(r["strike"] - spot) <= radius
   ]


   if not nearby:
       return {
           "regime": "UNKNOWN",
           "strongest_positive": None,
           "strongest_negative": None,
           "nearest_positive_above": None,
           "nearest_negative_below": None,
       }


   positive = max(
       nearby,
       key=lambda r: r.get("net_gex", 0),
       default=None,
   )


   negative = min(
       nearby,
       key=lambda r: r.get("net_gex", 0),
       default=None,
   )


   above_positive = [
       r for r in nearby
       if r["strike"] > spot and r.get("net_gex", 0) > 0
   ]


   below_negative = [
       r for r in nearby
       if r["strike"] < spot and r.get("net_gex", 0) < 0
   ]


   nearest_positive_above = (
       min(above_positive, key=lambda r: r["strike"], default=None)
   )


   nearest_negative_below = (
       max(below_negative, key=lambda r: r["strike"], default=None)
   )


   net_gex = gex_result.get("net_gex", 0) or 0


   if net_gex < 0:
       regime = "NEGATIVE_GAMMA"
   else:
       regime = "POSITIVE_GAMMA"


   return {
       "regime": regime,
       "strongest_positive": positive,
       "strongest_negative": negative,
       "nearest_positive_above": nearest_positive_above,
       "nearest_negative_below": nearest_negative_below,
   }




def generate_market_commentary(
   gex_result,
   price_levels,
   spot,
   level_events=None,
    smc_context=None,
    multi_timeframe=None,
):
   """
   Generate human-readable running commentary using:


     - Net GEX
     - Gamma flip
     - Call wall
     - Put wall
     - Local GEX concentration
     - PDH / PDL
     - PWH / PWL
     - Session levels
     - VWAP
     - Level state events
    - SMC trend strength and AMD phase


   This is structural commentary, not a trade recommendation.
   """


   comments = []


   net_gex = gex_result.get("net_gex", 0) or 0
   call_wall = gex_result.get("call_wall")
   put_wall = gex_result.get("put_wall")
   flip = gex_result.get("flip_level")


   # --------------------------------------------------------
   # GEX REGIME
   # --------------------------------------------------------


   if net_gex < 0:
       comments.append(
           f"NEGATIVE GAMMA: net GEX {_fmt_b(net_gex)}. "
           f"Expect moves to amplify rather than automatically mean-revert."
       )
   elif net_gex > 0:
       comments.append(
           f"POSITIVE GAMMA: net GEX {_fmt_b(net_gex)}. "
           f"Dealers may dampen moves and price may gravitate around key strikes."
       )


   # --------------------------------------------------------
   # GAMMA FLIP
   # --------------------------------------------------------


   if flip is not None:


       if spot > flip:
           comments.append(
               f"SPX is {_fmt_price(spot - flip)} above gamma flip "
               f"{_fmt_price(flip)}."
           )


       elif spot < flip:
           comments.append(
               f"SPX is {_fmt_price(flip - spot)} below gamma flip "
               f"{_fmt_price(flip)}."
           )


       else:
           comments.append(
               f"SPX is sitting directly on gamma flip {_fmt_price(flip)}."
           )


   # --------------------------------------------------------
   # CALL / PUT WALL
   # --------------------------------------------------------


   if call_wall is not None:


       distance = call_wall - spot


       if distance > 0:
           comments.append(
               f"CALL WALL {call_wall:,.0f} is "
               f"{distance:.1f} points above spot."
           )
       else:
           comments.append(
               f"SPX is above CALL WALL {call_wall:,.0f}; "
               f"watch for acceptance or failed retest."
           )


   if put_wall is not None:


       distance = spot - put_wall


       if distance > 0:
           comments.append(
               f"PUT WALL {put_wall:,.0f} is "
               f"{distance:.1f} points below spot."
           )
       else:
           comments.append(
               f"SPX is below PUT WALL {put_wall:,.0f}; "
               f"watch for continuation or reclaim."
           )


   # --------------------------------------------------------
   # LOCAL GEX
   # --------------------------------------------------------


   ctx = get_gex_context(
       gex_result,
       spot,
       radius=60,
   )


   strongest_negative = ctx.get("strongest_negative")
   strongest_positive = ctx.get("strongest_positive")


   if strongest_negative:
       comments.append(
           f"Largest nearby negative GEX is "
           f"{strongest_negative['strike']:,.0f} "
           f"({_fmt_b(strongest_negative['net_gex'])})."
       )


   if strongest_positive:
       comments.append(
           f"Largest nearby positive GEX is "
           f"{strongest_positive['strike']:,.0f} "
           f"({_fmt_b(strongest_positive['net_gex'])})."
       )


   # --------------------------------------------------------
   # PRICE LEVELS
   # --------------------------------------------------------


   levels = price_levels.get("levels", []) if price_levels else []


   above = _nearest_level(levels, spot, "above")
   below = _nearest_level(levels, spot, "below")


   if above:
       comments.append(
           f"Nearest price level above: "
           f"{above['name']} {_fmt_price(above['price'])}."
       )


   if below:
       comments.append(
           f"Nearest price level below: "
           f"{below['name']} {_fmt_price(below['price'])}."
       )


   # --------------------------------------------------------
   # SMC TREND / AMD CONTEXT
   # --------------------------------------------------------


   smc = smc_context or {}
   if smc:
       bias = smc.get("bias", "NEUTRAL")
       strength = smc.get("trend_strength", "NEUTRAL")
       score = smc.get("trend_strength_score", 0)
       comments.append(f"SMC TREND: {strength} {bias} ({score}/100).")

       weakness_level = smc.get("weakness_level")
       weakness_condition = smc.get("weakness_condition")
       if weakness_level is not None and weakness_condition:
           comments.append(
               f"Trend weakness / invalidation reference: a close "
               f"{weakness_condition.lower()} {_fmt_price(weakness_level)}."
           )
       weakness_signals = smc.get("weakness_signals") or []
       if weakness_signals:
           comments.append("Weakness evidence: " + "; ".join(weakness_signals) + ".")

       phase = smc.get("amd_phase", "UNCONFIRMED")
       phase_direction = smc.get("amd_direction")
       phase_label = f" {phase_direction}" if phase_direction else ""
       phase_evidence = smc.get("amd_evidence")
       comments.append(
           f"AMD: {phase}{phase_label}"
           + (f" — {phase_evidence}." if phase_evidence else ".")
       )

   timeframe_context = (multi_timeframe or {}).get("timeframes") or {}
   if timeframe_context:
       labels = ("15M", "1H", "4H", "DAY", "WEEKLY")
       bias_text = ", ".join(
           f"{label} {(timeframe_context.get(label) or {}).get('bias', 'NEUTRAL')}"
           for label in labels if label in timeframe_context
       )
       comments.append(
           f"Multi-timeframe structure: {bias_text}; aggregate "
           f"{(multi_timeframe or {}).get('htf_bias', 'NEUTRAL')}."
       )


   # --------------------------------------------------------
   # LEVEL EVENTS
   # --------------------------------------------------------


   for event in level_events or []:
       comments.append(event)


   # --------------------------------------------------------
   # ACTIONABLE STRUCTURAL COMMENTARY
   # --------------------------------------------------------


   if net_gex < 0:


       if flip is not None and spot < flip:
           comments.append(
               f"STRUCTURE: below gamma flip. "
               f"Breakouts can extend and downside moves can accelerate."
           )


       elif flip is not None and spot > flip:
           comments.append(
               f"STRUCTURE: above gamma flip but still negative GEX. "
               f"Watch whether price can hold above the flip."
           )


   return comments
# ============================================================
# LEVEL CLAIM / REJECTION ENGINE
# ============================================================


def update_level_states(
   levels,
   spot,
   previous_spot,
   state_store,
   tolerance=2.0,
   confirmation_distance=2.0,
):
   """
   Determines whether a level is:


       ABOVE
       BELOW
       TESTING
       CLAIM_CANDIDATE
       CLAIMED
       REJECTED
       LOST


   Important:
       A single touch does NOT equal rejection.
       A single cross does NOT equal claim.


   A CLAIM requires:
       1. Cross
       2. Retest
       3. Move away while holding the level


   A REJECTION requires:
       1. Test
       2. Move away from the level
       3. Failure to establish on the other side
   """


   events = []


   if spot is None:
       return events


   for level in levels:


       name = level.get("name")
       price = level.get("price")


       if not name or price is None:
           continue


       key = f"{name}:{round(price, 2)}"


       state = state_store.setdefault(
           key,
           {
               "state": "UNKNOWN",
               "direction": None,
               "tested": False,
               "retest": False,
               "last_spot": None,
           },
       )


       previous_state = state["state"]


       distance = spot - price
       abs_distance = abs(distance)


       # ----------------------------------------------------
       # INITIAL STATE
       # ----------------------------------------------------


       if state["last_spot"] is None:


           state["state"] = (
               "ABOVE" if spot > price
               else "BELOW"
           )


           state["last_spot"] = spot
           continue


       # ----------------------------------------------------
       # TESTING
       # ----------------------------------------------------


       if abs_distance <= tolerance:


           if not state["tested"]:


               state["tested"] = True
               state["state"] = "TESTING"


               events.append(
                   f"{name} {price:.2f} is being TESTED "
                   f"(SPX {spot:.2f})."
               )


       # ----------------------------------------------------
       # CROSS UP
       # ----------------------------------------------------


       if (
           previous_spot is not None
           and previous_spot < price - tolerance
           and spot >= price + confirmation_distance
       ):


           state["direction"] = "UP"
           state["state"] = "CLAIM_CANDIDATE"
           state["retest"] = False


           events.append(
               f"SPX crossed ABOVE {name} {price:.2f}. "
               f"Waiting for retest before calling it CLAIMED."
           )


       # ----------------------------------------------------
       # CROSS DOWN
       # ----------------------------------------------------


       elif (
           previous_spot is not None
           and previous_spot > price + tolerance
           and spot <= price - confirmation_distance
       ):


           state["direction"] = "DOWN"
           state["state"] = "CLAIM_CANDIDATE"
           state["retest"] = False


           events.append(
               f"SPX crossed BELOW {name} {price:.2f}. "
               f"Waiting for retest before calling it LOST."
           )


       # ----------------------------------------------------
       # RETEST AFTER UPWARD BREAK
       # ----------------------------------------------------


       if state["direction"] == "UP":


           if abs_distance <= tolerance:


               state["retest"] = True
               state["state"] = "RETESTING"


           elif state["retest"] and spot > price + confirmation_distance:


               state["state"] = "CLAIMED"


               events.append(
                   f"CLAIMED: SPX held {name} {price:.2f} "
                   f"on retest and moved back above."
               )


               state["tested"] = False
               state["retest"] = False


           elif state["retest"] and spot < price - confirmation_distance:


               state["state"] = "REJECTED"


               events.append(
                   f"REJECTED: SPX broke above {name} {price:.2f} "
                   f"but failed the retest."
               )


               state["tested"] = False
               state["retest"] = False


       # ----------------------------------------------------
       # RETEST AFTER DOWNWARD BREAK
       # ----------------------------------------------------


       elif state["direction"] == "DOWN":


           if abs_distance <= tolerance:


               state["retest"] = True
               state["state"] = "RETESTING"


           elif state["retest"] and spot < price - confirmation_distance:


               state["state"] = "LOST"


               events.append(
                   f"LOST: SPX failed to reclaim {name} "
                   f"{price:.2f} after the retest."
               )


               state["tested"] = False
               state["retest"] = False


           elif state["retest"] and spot > price + confirmation_distance:


               state["state"] = "CLAIMED"


               events.append(
                   f"RECLAIMED: SPX broke below {name} "
                   f"{price:.2f} but recovered above it."
               )


               state["tested"] = False
               state["retest"] = False


       # ----------------------------------------------------
       # SIMPLE REJECTION WITHOUT A BREAK
       # ----------------------------------------------------


       if (
           state["tested"]
           and state["direction"] is None
           and previous_spot is not None
       ):


           # approached from below and moved back down
           if (
               previous_spot <= price
               and spot < price - confirmation_distance
           ):


               state["state"] = "REJECTED"


               events.append(
                   f"REJECTED: {name} {price:.2f}. "
                   f"SPX tested the level and moved back lower."
               )


               state["tested"] = False


           # approached from above and moved back up
           elif (
               previous_spot >= price
               and spot > price + confirmation_distance
           ):


               state["state"] = "REJECTED"


               events.append(
                   f"REJECTED: {name} {price:.2f}. "
                   f"SPX tested the level and moved back higher."
               )


               state["tested"] = False


       state["last_spot"] = spot


   return events
# ============================================================
# LEVEL STATE ENGINE
# ============================================================


def collect_important_levels(
   gex: Dict[str, Any],
   price_levels: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
   """Merge gamma-derived levels and price-structure levels into one named
   list, keyed the same way update_level_states expects (name + price)."""
   levels: List[Dict[str, Any]] = []


   def add(name: str, price: Optional[float], kind: str,
           extra: Optional[Dict[str, Any]] = None) -> None:
       if price is None:
           return
       entry = {"name": name, "price": num(price), "kind": kind}
       if extra:
           entry.update(extra)
       levels.append(entry)


   gamma_levels = gex.get("gamma_levels") or {}
   upper = gamma_levels.get("upper_gamma_wall")
   lower = gamma_levels.get("lower_gamma_wall")


   add("GAMMA ABOVE", gamma_level_strike(upper), "gamma",
       {"dominant_side": upper["dominant_side"], "net_gex": upper["net_gex"]} if upper else None)
   add("GAMMA BELOW", gamma_level_strike(lower), "gamma",
       {"dominant_side": lower["dominant_side"], "net_gex": lower["net_gex"]} if lower else None)
   add("CALL WALL", gamma_level_strike(gamma_levels.get("call_wall_above")), "gamma",
       {"dominant_side": "call"})
   add("PUT WALL", gamma_level_strike(gamma_levels.get("put_wall_below")), "gamma",
       {"dominant_side": "put"})


   transition = gex.get("local_gamma_transition")
   add("GAMMA FLIP", transition["interpolated"] if transition else None, "flip")


   for level in (price_levels or {}).get("levels", []):
       add(level["name"], level.get("price"), level.get("kind", "price"))


   return levels




def build_level_state_rows(
   levels: List[Dict[str, Any]],
   spot: Optional[float],
   state_store: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
   """LEVEL / DISTANCE / STATE table, deduped by name+price and sorted
   highest to lowest so neighboring rows are trade-plan neighbors."""
   if spot is None:
       return []


   rows = []
   seen = set()
   for level in levels:
       name, price = level.get("name"), level.get("price")
       if not name or price is None:
           continue
       key = f"{name}:{round(price, 2)}"
       if key in seen:
           continue
       seen.add(key)


       info = state_store.get(key) or {}
       engine_state = info.get("state")
       # update_level_states's plain ABOVE/BELOW describe spot relative to
       # the level (untouched state); the level table instead describes the
       # level relative to spot, so only pass through the meaningful states.
       if engine_state in (None, "ABOVE", "BELOW", "UNKNOWN"):
           display_state = "ABOVE" if price > spot else "BELOW"
       else:
           display_state = engine_state


       rows.append({
           "name": name,
           "price": price,
           "distance": price - spot,
           "state": display_state,
           "kind": level.get("kind", "price"),
           "dominant_side": level.get("dominant_side"),
       })


   rows.sort(key=lambda r: r["price"], reverse=True)
   return rows




_ACTIVE_LEVEL_STATES = {"TESTING", "CLAIM_CANDIDATE", "RETESTING"}




def generate_level_commentary(
   rows: List[Dict[str, Any]],
   net_gex: float,
) -> List[str]:
   """
   Running commentary for whichever level is currently in play, e.g.:


     "SPX testing 7650. Negative gamma dominates below. A 5-min close
     above 7650 followed by successful retest would claim 7650 and open
     7675. Failure back below 7650 keeps 7630 in play."


   Neighbors come from the sorted level table, so "open X" / "keeps Y in
   play" always point at the next real level, not a fixed offset.
   """
   comments = []
   gamma_phrase = (
       "Negative gamma dominates below" if net_gex < 0
       else "Positive gamma may cushion moves"
   )


   for i, row in enumerate(rows):
       state = row["state"]
       name, price = row["name"], row["price"]
       above_row = rows[i - 1] if i > 0 else None
       below_row = rows[i + 1] if i + 1 < len(rows) else None
       label = f"major negative-GEX {name}" if row.get("dominant_side") == "put" else name


       if state in _ACTIVE_LEVEL_STATES:
           open_target = f"{above_row['price']:.0f}" if above_row else "the next level"
           fallback = f"{below_row['price']:.0f}" if below_row else "the next level below"
           comments.append(
               f"SPX testing {label} {price:.0f}. {gamma_phrase}. "
               f"A 5-min close above {price:.0f} followed by successful "
               f"retest would claim {price:.0f} and open {open_target}. "
               f"Failure back below {price:.0f} keeps {fallback} in play."
           )
       elif state == "CLAIMED":
           open_target = f"{above_row['price']:.0f}" if above_row else "higher levels"
           comments.append(f"SPX claimed {name} {price:.0f}; {open_target} is next in play.")
       elif state in ("REJECTED", "LOST"):
           fallback = f"{below_row['price']:.0f}" if below_row else "lower levels"
           comments.append(f"SPX rejected at {name} {price:.0f}; downside continuation opens {fallback}.")


   return comments




if __name__ == "__main__":
   print(
       "Module ready. Connect your Tradier chain to:\n"
       "    result = analyze_chain(chain, spot)\n"
       "    print(format_analysis(result))"
   )
