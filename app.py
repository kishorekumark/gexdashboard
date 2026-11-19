import csv
import json
import os
import time
import threading
from collections import deque
from datetime import date, datetime, timedelta
from math import sqrt
from pathlib import Path
from zoneinfo import ZoneInfo


import requests
from flask import Flask, abort, jsonify, render_template, request, send_file
from tradier_client import TradierClient
from gex_engine import (
   compute_gex,
   compute_max_pain,
   detect_events,
   compute_volume_meter,
   compute_leaderboards,
   compute_dex,
   compute_vex,
   compute_flow,
   compute_live_flow,
   build_options_heatmap,
   build_dealer_supply_demand,
   build_trade_idea,
   generate_market_commentary,
   generate_trade_plan_events,
   gamma_level_strike,
   collect_important_levels,
   build_level_state_rows,
   generate_level_commentary,
   update_level_states,
)
from price_levels import (
    build_multi_timeframe_context,
    build_price_levels,
    compute_expected_move_calibration,
    compute_smc_context,
    normalize_bars,
)


LEADERBOARD_SIZE = int(os.environ.get("LEADERBOARD_SIZE", "5"))
MAX_EVENTS = int(os.environ.get("MAX_EVENTS", "200"))
MAX_TRADE_HISTORY = int(os.environ.get("MAX_TRADE_HISTORY", "100"))
MAX_FLOW_EVENTS = int(os.environ.get("MAX_FLOW_EVENTS", "500"))
TRADE_MAJOR_SHIFT_POINTS = float(os.environ.get("TRADE_MAJOR_SHIFT_POINTS", "5"))
TRADIER_TOKEN = os.environ.get("TRADIER_TOKEN")
SANDBOX = os.environ.get("TRADIER_SANDBOX", "false").lower() == "true"
UNDERLYING_SYMBOL = os.environ.get("UNDERLYING_SYMBOL", "SPX")
OPTION_SYMBOL = os.environ.get("OPTION_SYMBOL", "SPX")
EXPIRATION = os.environ.get("EXPIRATION")
POLL_SECONDS = max(1, int(os.environ.get("POLL_SECONDS", "30")))
STRIKE_WINDOW = float(os.environ.get("STRIKE_WINDOW", "40"))
FLOW_DISPLAY_STRIKES = 20
# Gamma levels are only searched within this many points of spot so a far
# out-of-the-money strike cannot become the reference intraday level.
GAMMA_LEVEL_WINDOW = float(os.environ.get("GAMMA_LEVEL_WINDOW", "50"))
PRICE_BAR_INTERVAL = os.environ.get("PRICE_BAR_INTERVAL", "5min")
PRICE_HISTORY_DAYS = int(os.environ.get("PRICE_HISTORY_DAYS", "90"))
PRICE_REFRESH_SECONDS = max(30, int(os.environ.get("PRICE_REFRESH_SECONDS", "60")))
VIX_SYMBOL = os.environ.get("VIX_SYMBOL", "VIX")
PORT = int(os.environ.get("PORT", "5000"))
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).with_name("data")))
app = Flask(__name__)
_lock = threading.Lock()
_latest = {
   "spot": None, "net_gex": None, "gross_call_gex": None, "gross_put_gex": None,
   "call_wall": None, "put_wall": None, "flip_level": None,
   "upper_gamma_wall": None, "lower_gamma_wall": None,
   "call_wall_above": None, "put_wall_below": None,
    "gamma_levels": {}, "price_levels": [], "price_structure": {}, "price_bars": [],
    "price_session_ohlc": None,
   "local_gamma_transition": None, "max_pain": None, "table": [],
   "volume_meter": {"call_volume": 0, "put_volume": 0, "put_call_ratio": None, "call_pct": 50.0},
   "leaders": {"call_oi": [], "put_oi": [], "call_volume": [], "put_volume": []},
    "live_flow": {"events": [], "totals": {}, "directional_score": 0.0,
                      "classification_coverage_pct": 0.0, "source": "volume increments"},
    "options_heatmap": [],
    "dealer_supply_demand": [],
   "net_dex": None, "dex_table": [], "total_call_vega": None,
    "total_put_vega": None, "vex_table": [], "trade_plan": [], "trade_idea": {},
   "level_states": [], "commentary": [],
    "vix": None, "vix_source": None, "atm_straddle": None,
    "session_playbook": {}, "updated_at": None, "error": None,
}
_events = []
_trade_history = []
_last_trade_idea = None
_last_trade_event_state = {}
_last_structural_events = set()
# Keyed by "NAME:price" -> claim/rejection state, persists across polls.
_level_state_store = {}
_gex_baseline = None


def _build_gex_change(current_table, baseline_table, baseline_time, spot, has_comparison):
   baseline_by_strike = {float(row["strike"]): row for row in baseline_table}
   current_by_strike = {float(row["strike"]): row for row in current_table}
   rows = []
   for strike in sorted(set(baseline_by_strike) | set(current_by_strike)):
       if spot is not None and abs(strike - spot) > STRIKE_WINDOW:
           continue
       baseline = baseline_by_strike.get(strike, {})
       current = current_by_strike.get(strike, {})
       baseline_call = float(baseline.get("call_gex") or 0.0)
       baseline_put = abs(float(baseline.get("put_gex") or 0.0))
       current_call = float(current.get("call_gex") or 0.0)
       current_put = abs(float(current.get("put_gex") or 0.0))
       rows.append({
           "strike": strike,
           "call_change": current_call - baseline_call,
           "put_change": current_put - baseline_put,
           "call_current": current_call,
           "put_current": current_put,
           "call_baseline": baseline_call,
           "put_baseline": baseline_put,
       })
   return {
       "baseline_time": baseline_time,
       "has_comparison": has_comparison,
       "rows": rows,
   }


class _LiveFlowTracker:
   """Track volume increments from successive Tradier chain snapshots."""

   def __init__(self):
       self.previous = {}
       self.events = deque(maxlen=MAX_FLOW_EVENTS)
       self.cvd_by_strike = {}
       self.cvd_history = deque(maxlen=240)
       self.initialized = False

   def update(self, chain, spot):
       current = {
           (opt.get("symbol") or f"{opt.get('option_type')}:{opt.get('strike')}"): int(float(opt.get("volume") or 0))
           for opt in chain
       }
       if not self.initialized:
           self.previous = current
           self.initialized = True
           return {"events": [], "totals": {}, "directional_score": 0.0,
                   "classification_coverage_pct": 0.0,
                   "baseline_ready": False,
                   "cvd": {"series": [], "strike_table": []}}, []

       flow = compute_live_flow(self.previous, chain)
       flow["baseline_ready"] = True
       self.previous = current
       now = datetime.now().strftime("%H:%M:%S")
       for event in flow["events"]:
           strike = float(event["strike"])
           row = self.cvd_by_strike.setdefault(strike, {
               "strike": strike, "calls_net": 0, "puts_net": 0,
               "calls_net_dollars": 0.0, "puts_net_dollars": 0.0,
               "calls_buy": 0, "calls_sell": 0, "puts_buy": 0, "puts_sell": 0,
           })
           signed_contracts = event["contracts"] if event["side"] == "BUY" else -event["contracts"] if event["side"] == "SELL" else 0
           signed_dollars = signed_contracts * float(event.get("last") or 0) * 100
           if event["option_type"] == "call":
               row["calls_net"] += signed_contracts
               row["calls_net_dollars"] += signed_dollars
               if event["side"] in {"BUY", "SELL"}:
                   row[f"calls_{event['side'].lower()}"] += event["contracts"]
           else:
               row["puts_net"] += signed_contracts
               row["puts_net_dollars"] += signed_dollars
               if event["side"] in {"BUY", "SELL"}:
                   row[f"puts_{event['side'].lower()}"] += event["contracts"]
           self.events.append({**event, "time": now})
       calls_net = sum(row["calls_net"] for row in self.cvd_by_strike.values())
       puts_net = sum(row["puts_net"] for row in self.cvd_by_strike.values())
       calls_dollars = sum(row["calls_net_dollars"] for row in self.cvd_by_strike.values())
       puts_dollars = sum(row["puts_net_dollars"] for row in self.cvd_by_strike.values())
       self.cvd_history.append({
           "time": now, "spot": spot, "calls_net": calls_net, "puts_net": puts_net,
           "dollars_net": calls_dollars + puts_dollars,
           "calls_flow_premium": sum(
               (1 if event["side"] == "BUY" else -1 if event["side"] == "SELL" else 0)
               * float(event.get("contracts") or 0) * float(event.get("last") or 0) * 100
               for event in flow["events"] if event["option_type"] == "call"
           ),
           "puts_flow_premium": sum(
               (1 if event["side"] == "BUY" else -1 if event["side"] == "SELL" else 0)
               * float(event.get("contracts") or 0) * float(event.get("last") or 0) * 100
               for event in flow["events"] if event["option_type"] == "put"
           ),
           "calls_drift": calls_dollars,
           "puts_drift": puts_dollars,
       })
       recent = list(self.events)
       flow["events"] = recent
       flow["cvd"] = {
           "series": list(self.cvd_history),
           "strike_table": sorted(self.cvd_by_strike.values(), key=lambda row: row["strike"]),
       }
       return flow, recent




def _expiration():
   return EXPIRATION or date.today().isoformat()




def _safe_float(value):
   try:
       return None if value is None else float(value)
   except (TypeError, ValueError):
       return None


def _format_price(value):
   parsed = _safe_float(value)
   return f"{parsed:.2f}" if parsed is not None else "-"


def _atm_straddle(chain, spot):
   contracts = {}
   for option in chain:
       kind = str(option.get("option_type") or "").lower()
       strike = _safe_float(option.get("strike"))
       bid = _safe_float(option.get("bid"))
       ask = _safe_float(option.get("ask"))
       if kind not in ("call", "put") or strike is None or bid is None or ask is None:
           continue
       if bid < 0 or ask <= 0 or ask < bid:
           continue
       greeks = option.get("greeks") or {}
       contracts.setdefault(strike, {})[kind] = {
           "mid": (bid + ask) / 2,
           "theta": _safe_float(greeks.get("theta")),
       }

   paired = [strike for strike, sides in contracts.items() if "call" in sides and "put" in sides]
   if not paired:
       return None

   strike = min(paired, key=lambda value: abs(value - spot))
   call, put = contracts[strike]["call"], contracts[strike]["put"]
   theta = call["theta"] + put["theta"] if call["theta"] is not None and put["theta"] is not None else None
   return {
       "strike": strike,
       "call_mid": call["mid"],
       "put_mid": put["mid"],
       "premium": call["mid"] + put["mid"],
       "theta_per_day": theta,
       "quote_source": "bid/ask midpoint",
   }


def _build_session_playbook(
   spot,
   net_gex,
   gamma_levels,
   flip,
   price_structure,
   vix,
   vix_source,
   atm_straddle,
   daily_bars,
   vix_daily_bars,
   intraday_bars,
   as_of=None,
   now=None,
):
   now = now or datetime.now(ZoneInfo("America/New_York"))
   as_of = as_of or now.date()
   previous_day = (price_structure or {}).get("prior_day") or {}
   reference = _safe_float(previous_day.get("close")) or _safe_float(spot)
   volatility = _safe_float(vix)
   expected_move = reference * volatility / (100 * sqrt(252)) if reference and volatility else None
   weekly_move = reference * volatility * sqrt(5 / 252) / 100 if reference and volatility else None
   calibration = compute_expected_move_calibration(
       daily_bars, vix_daily_bars, expected_move, as_of=as_of, lookback=60,
   )

   lower_shelf = reference - expected_move * 0.5 if expected_move is not None else None
   upper_shelf = reference + expected_move * 0.5 if expected_move is not None else None
   lower_band = reference - expected_move if expected_move is not None else None
   upper_band = reference + expected_move if expected_move is not None else None
   put_wall = gamma_level_strike((gamma_levels or {}).get("put_wall_below"))
   call_wall = gamma_level_strike((gamma_levels or {}).get("call_wall_above"))
   vwap = _safe_float((price_structure or {}).get("vwap"))
   net_gex = _safe_float(net_gex) or 0.0
   negative_gamma = net_gex < 0
   fast_regime = negative_gamma and volatility is not None and volatility > 18.5

   levels = []
   def add_level(name, value, source):
       value = _safe_float(value)
       if value is not None:
           levels.append({"name": name, "price": value, "distance": value - spot, "source": source})

   add_level("Prior close", reference, "Daily close")
   add_level("Lower full move", lower_band, "VIX 1-day band")
   add_level("Lower half move", lower_shelf, "Half expected move")
   for level in ((price_structure or {}).get("levels") or []):
       if level.get("name") in ("PDH", "PDL", "PDC", "VWAP"):
           add_level(level["name"], level.get("price"), level.get("kind", "price structure"))
   add_level("Put gamma wall", put_wall, "Current option-chain OI model")
   add_level("Gamma flip", flip, "Current option-chain OI model")
   add_level("Call gamma wall", call_wall, "Current option-chain OI model")
   add_level("Upper half move", upper_shelf, "Half expected move")
   add_level("Upper full move", upper_band, "VIX 1-day band")
   levels.sort(key=lambda level: level["price"], reverse=True)

   put_shelf_gap = abs(put_wall - lower_shelf) if put_wall is not None and lower_shelf is not None else None
   put_confluence = put_shelf_gap is not None and put_shelf_gap <= 5

   completed_bars = []
   uses_five_minute_bars = str(PRICE_BAR_INTERVAL).lower() in ("5min", "5m")
   for bar in normalize_bars(intraday_bars):
       stamp = bar["time"]
       if stamp.tzinfo is None:
           stamp = stamp.replace(tzinfo=ZoneInfo("America/New_York"))
       local = stamp.astimezone(ZoneInfo("America/New_York"))
       minute = local.hour * 60 + local.minute
       if local.date() != as_of or not (570 <= minute < 960):
           continue
       if uses_five_minute_bars and local + timedelta(minutes=5) <= now:
           completed_bars.append({"time": local, "close": bar["close"]})
   last_two_bars = completed_bars[-2:]
   consecutive_bars = (
       len(last_two_bars) == 2
       and last_two_bars[1]["time"] - last_two_bars[0]["time"] == timedelta(minutes=5)
   )
   last_two = [bar["close"] for bar in last_two_bars]
   after_opening_window = now.hour * 60 + now.minute >= 600
   downside_level = min(value for value in (lower_shelf, put_wall) if value is not None) if any(value is not None for value in (lower_shelf, put_wall)) else None
   upside_levels = [value for value in (upper_shelf, vwap) if value is not None]
   upside_level = max(upside_levels) if upside_levels else None
   downside_confirmed = (
       negative_gamma and after_opening_window and downside_level is not None
       and consecutive_bars and all(close < downside_level for close in last_two)
   )
   upside_confirmed = (
       uses_five_minute_bars and upside_level is not None and consecutive_bars
       and all(close > upside_level for close in last_two)
   )

   if fast_regime:
       regime = "FAST · negative GEX model + VIX above 18.5"
   elif negative_gamma:
       regime = "Negative GEX model · momentum risk elevated"
   elif net_gex > 0:
       regime = "Positive GEX model · mean-reversion tendency"
   else:
       regime = "Neutral GEX model"

   theta = _safe_float((atm_straddle or {}).get("theta_per_day"))
   straddle = _safe_float((atm_straddle or {}).get("premium"))
   return {
       "reference": reference,
       "vix": volatility,
       "vix_source": vix_source,
       "net_gex": net_gex,
       "gex_regime": regime,
       "expected_move": expected_move,
       "weekly_move": weekly_move,
    "morning_budget": expected_move * 0.555 if expected_move is not None else None,
       "lower_band": lower_band,
       "lower_shelf": lower_shelf,
       "upper_shelf": upper_shelf,
       "upper_band": upper_band,
       "calibration": calibration,
       "atm_straddle": atm_straddle,
       "straddle_to_move": straddle / expected_move if straddle is not None and expected_move else None,
       "theta_per_day": theta,
       "levels": levels,
       "put_shelf_gap": put_shelf_gap,
       "put_confluence": put_confluence,
       "scenarios": [
           {
               "name": "Downside acceptance",
               "status": "CONFIRMED" if downside_confirmed else "WAIT",
               "rule": (
                   f"Set PRICE_BAR_INTERVAL=5min to evaluate; current interval is {PRICE_BAR_INTERVAL}."
                   if not uses_five_minute_bars else
                   f"After 10:00 ET, two completed 5-minute closes below {downside_level:.2f}; "
                   "only a momentum-risk condition when the GEX model is negative."
                   if downside_level is not None else "Requires a valid lower shelf or put gamma wall."
               ),
               "invalid": "A close back above the broken level cancels acceptance.",
           },
           {
               "name": "Upside reclaim",
               "status": "CONFIRMED" if upside_confirmed else "WAIT",
               "rule": (
                   f"Set PRICE_BAR_INTERVAL=5min to evaluate; current interval is {PRICE_BAR_INTERVAL}."
                   if not uses_five_minute_bars else
                   f"Two completed 5-minute closes above {upside_level:.2f} "
                   "(upper half-move and VWAP when both are available)."
                   if upside_level is not None else "Requires a valid upper shelf or VWAP."
               ),
               "invalid": "A close back below the reclaimed level cancels acceptance.",
           },
           {
               "name": "No confirmation",
               "status": "WAIT" if not (downside_confirmed or upside_confirmed) else "ACTIVE",
               "rule": "Without a confirmed hold, treat breaks as unconfirmed; avoid assigning direction inside the band.",
               "invalid": "Reassess after two completed 5-minute closes beyond a trigger.",
           },
       ],
       "checklist": [
           "Compare the opening print with the prior-close reference and nearby levels.",
           "Confirm VIX and the current-expiration GEX model; OI does not reveal dealer ownership.",
           "Check the economic calendar separately; no calendar feed is connected.",
       ],
       "model_note": "OI-based SPX chain estimate, not observed dealer positions or orders; walls/flip are model-dependent.",
       "theta_note": "ATM straddle midpoint and chain theta are live estimates; fixed-time decay projections are not implied.",
       "morning_budget_note": "The 0.555 morning multiplier is the supplied post's hypothesis, not a validated forecast.",
   }
from collections import defaultdict, deque



def _trim_by_spot(rows, spot):
   if not rows or spot is None:
       return []
   result = [r for r in rows if _safe_float(r.get("strike")) is not None and abs(float(r["strike"]) - spot) <= STRIKE_WINDOW]
   return sorted(result, key=lambda r: r["strike"])


def _nearest_strikes(chain, spot, limit=FLOW_DISPLAY_STRIKES):
    strikes = {
         _safe_float(option.get("strike"))
         for option in chain
         if _safe_float(option.get("strike")) is not None
    }
    return set(sorted(strikes, key=lambda strike: (abs(strike - spot), strike))[:limit])


def _filter_live_flow_for_spot(flow, strikes):
    filtered = dict(flow)
    filtered["events"] = [
         event for event in flow.get("events", [])
         if _safe_float(event.get("strike")) in strikes
    ]
    cvd = dict(flow.get("cvd") or {})
    cvd["strike_table"] = [
         row for row in cvd.get("strike_table", [])
         if _safe_float(row.get("strike")) in strikes
    ]
    filtered["cvd"] = cvd
    filtered["display_strikes"] = sorted(strikes)
    return filtered




def _append_event(event):
   _events.append(event)
   del _events[:-MAX_EVENTS]




def _append_structural_events(messages, now_str):
   for message in messages or []:
       if not message:
           continue
       key = str(message)
       if key in _last_structural_events:
           continue
       _last_structural_events.add(key)
       # Keep dedup state bounded.
       if len(_last_structural_events) > MAX_EVENTS:
           _last_structural_events.clear()
       print(f"[{now_str}] EVENT: {message}")
       _append_event({
           "time": now_str, "category": "STRUCTURAL", "type": "STRUCTURAL",
           "severity": "info", "title": "STRUCTURAL EVENT", "message": str(message),
       })




def _append_trade_plan_events(events, now_str):
   for event in events or []:
       if not isinstance(event, dict):
           continue
       event_type = event.get("type") or "TRADE_PLAN"
       severity = event.get("severity") or "neutral"
       title = event.get("title") or event_type
       message = event.get("message") or ""
       direction = event.get("direction")
       trigger = event.get("trigger")
       state_key = (event_type, severity, title, message, direction, trigger)
       if _last_trade_event_state.get(event_type) == state_key:
           continue
       _last_trade_event_state[event_type] = state_key
       print(f"[{now_str}] TRADE PLAN: {title} - {message}")
       _append_event({
           "time": now_str, "category": "TRADE_PLAN", "type": event_type,
           "severity": severity, "title": title, "message": message,
           "direction": direction, "trigger": trigger,
       })


def _record_trade_idea(idea, now_str):
   """Append only major trade-idea changes, ignoring normal entry drift."""
   global _last_trade_idea
   if not isinstance(idea, dict) or not idea.get("action"):
       return
   if _last_trade_idea is not None:
       previous = _last_trade_idea
       def price_shift(field):
           current = _safe_float(idea.get(field))
           prior = _safe_float(previous.get(field))
           return abs(current - prior) if current is not None and prior is not None else 0.0
       structural_shift = any(
           idea.get(field) != previous.get(field)
           for field in (
               "action", "smc_bias", "bos", "liquidity_path", "active_fvg",
               "inverse_fvg", "timeframe_biases", "timeframe_alignment",
           )
       )
       level_shift = any(
           price_shift(field) >= TRADE_MAJOR_SHIFT_POINTS
           for field in ("tp1", "tp2", "tp3", "stop_loss")
       )
       if not structural_shift and not level_shift:
           return

   _last_trade_idea = dict(idea)
   _trade_history.append({**idea, "time": now_str})
   del _trade_history[:-MAX_TRADE_HISTORY]
   print(
       f"[{now_str}] TRADE IDEA HISTORY: {idea['action']} "
       f"entry={idea.get('entry')} target={idea.get('target')} "
       f"stop={idea.get('stop_loss')}"
   )




def _build_summary(spot, gex, dex, price_structure):
   # call_wall/put_wall are retained for compatibility, but the trade plan is
   # driven by the gamma concentrations above and below spot.
   levels = gex.get("gamma_levels") or {}
   return {
       "spot": spot,
       "net_gex": gex.get("net_gex", 0.0),
       "call_wall": gex.get("call_wall"),
       "put_wall": gex.get("put_wall"),
       "upper_gamma_wall": gamma_level_strike(levels.get("upper_gamma_wall")),
       "lower_gamma_wall": gamma_level_strike(levels.get("lower_gamma_wall")),
       "call_wall_above": gamma_level_strike(levels.get("call_wall_above")),
       "put_wall_below": gamma_level_strike(levels.get("put_wall_below")),
       "flip_level": gex.get("flip_level"),
       "local_gamma_transition": gex.get("local_gamma_transition"),
       "net_dex": dex.get("net_dex", 0.0),
    "flow_score": 0.0,
    "smc": (price_structure or {}).get("smc", {}),
         "multi_timeframe": (price_structure or {}).get("multi_timeframe", {}),
       "price_levels": (price_structure or {}).get("levels", []),
   }


def _build_contract_statistics(chain, live_events):
   totals = {
       "calls": {"premium": 0.0, "volume": 0, "events": 0},
       "puts": {"premium": 0.0, "volume": 0, "events": 0},
   }
   for option in chain:
       option_type = str(option.get("option_type") or "").lower()
       kind = "calls" if option_type == "call" else "puts" if option_type == "put" else ""
       if kind not in totals:
           continue
       try:
           volume = max(0, int(float(option.get("volume") or 0)))
           last = max(0.0, float(option.get("last") or 0))
       except (TypeError, ValueError):
           volume, last = 0, 0.0
       totals[kind]["volume"] += volume
       totals[kind]["premium"] += volume * last * 100

   for event in live_events:
       option_type = str(event.get("option_type") or "").lower()
       kind = "calls" if option_type == "call" else "puts" if option_type == "put" else ""
       if kind in totals:
           totals[kind]["events"] += 1
   return totals


def _append_daily_csv(dataset, fields, rows, export_date):
   if not rows:
       return
   path = DATA_DIR / f"{export_date}_{dataset}.csv"
   path.parent.mkdir(parents=True, exist_ok=True)
   write_header = not path.exists() or path.stat().st_size == 0
   with path.open("a", newline="", encoding="utf-8") as output:
       writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
       if write_header:
           writer.writeheader()
       writer.writerows(rows)


def _persist_daily_data(spot, expiration, gex_table, flow_events):
   export_date = date.today().isoformat()
   timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
   gex_fields = [
       "timestamp", "underlying", "option_symbol", "expiration", "spot",
       "strike", "call_gex", "put_gex", "net_gex", "call_oi", "put_oi",
       "call_volume", "put_volume",
   ]
   gex_rows = [
       {
           "timestamp": timestamp,
           "underlying": UNDERLYING_SYMBOL,
           "option_symbol": OPTION_SYMBOL,
           "expiration": expiration,
           "spot": spot,
           **row,
       }
       for row in gex_table
   ]
   flow_fields = [
       "timestamp", "underlying", "option_symbol", "expiration", "spot",
       "symbol", "option_type", "strike", "side", "contracts", "last", "bid", "ask",
   ]
   flow_rows = [
       {
           **event,
           "timestamp": timestamp,
           "underlying": UNDERLYING_SYMBOL,
           "option_symbol": OPTION_SYMBOL,
           "expiration": event.get("expiration") or expiration,
           "spot": spot,
       }
       for event in flow_events
   ]
   _append_daily_csv("gex", gex_fields, gex_rows, export_date)
   _append_daily_csv("options_flow", flow_fields, flow_rows, export_date)




def _chart_bars(bars, limit=180):
   """Trim intraday bars to the most recent session and JSON-serializable fields."""
   normalized = normalize_bars(bars)
   if not normalized:
       return []
   last_day = normalized[-1]["time"].date()
   today = [b for b in normalized if b["time"].date() == last_day] or normalized
   return [
       {
           "time": b["time"].isoformat(),
           "open": b["open"],
           "high": b["high"],
           "low": b["low"],
           "close": b["close"],
           "volume": b["volume"],
       }
       for b in today[-limit:]
   ]


def _session_ohlc(bars):
   normalized = normalize_bars(bars)
   if not normalized:
       return None
   session_date = normalized[-1]["time"].date()
   session = [bar for bar in normalized if bar["time"].date() == session_date]
   if not session:
       return None
   return {
       "date": session_date.isoformat(),
       "open": session[0]["open"],
       "high": max(bar["high"] for bar in session),
       "low": min(bar["low"] for bar in session),
       "close": session[-1]["close"],
   }




class _PriceFeed:
   """Fetches daily and intraday bars, refreshed on its own slower cadence."""


   def __init__(self, client, symbol):
             self.client = client
             self.symbol = symbol
             self.daily = []
             self.vix_daily = []
             self.intraday = []
             self.hourly = []
             self.bars_15m = []
             self.bars_1m = []
             self._fetched_at = 0.0


   def refresh(self, force=False):
       if not force and time.time() - self._fetched_at < PRICE_REFRESH_SECONDS:
           return self.daily, self.intraday


       end = date.today()
       start = end - timedelta(days=PRICE_HISTORY_DAYS)
       self.daily = self.client.get_history(
           self.symbol, interval="daily",
           start=start.isoformat(), end=end.isoformat(),
       )
       try:
           self.vix_daily = self.client.get_history(
               VIX_SYMBOL, interval="daily",
               start=start.isoformat(), end=end.isoformat(),
           )
       except Exception as exc:
           print(f"VIX HISTORY ERROR ({VIX_SYMBOL}): {exc}")
       self.intraday = self.client.get_timesales(
           self.symbol, interval=PRICE_BAR_INTERVAL,
           start=(end - timedelta(days=3)).isoformat(), end=end.isoformat(),
       )
       self.hourly = self.client.get_timesales(
           self.symbol, interval="1h",
           start=(end - timedelta(days=20)).isoformat(), end=end.isoformat(),
       )
       self.bars_15m = self.client.get_timesales(
           self.symbol, interval="15min",
           start=(end - timedelta(days=10)).isoformat(), end=end.isoformat(),
       )
       self.bars_1m = self.client.get_timesales(
           self.symbol, interval="1min",
           start=(end - timedelta(days=3)).isoformat(), end=end.isoformat(),
       )
       self._fetched_at = time.time()
       return self.daily, self.intraday




def _poll_loop():
    global _gex_baseline
    client = TradierClient(token=TRADIER_TOKEN, sandbox=SANDBOX)
    price_feed = _PriceFeed(client, UNDERLYING_SYMBOL)
    flow_tracker = _LiveFlowTracker()
    prev_summary = None
    prev_spot = None
    expiration = _expiration()


    while True:
       now_str = datetime.now().strftime("%H:%M:%S")
       try:
           quote = client.get_quote(UNDERLYING_SYMBOL)
           if not quote or quote.get("last") is None:
               raise RuntimeError(f"No quote for {UNDERLYING_SYMBOL}")
           spot = float(quote["last"])

           vix_value = None
           vix_source = "unavailable"
           try:
               vix_quote = client.get_quote(VIX_SYMBOL)
               if vix_quote and vix_quote.get("last") is not None:
                   vix_value = float(vix_quote["last"])
                   vix_source = f"{VIX_SYMBOL} live quote"
           except Exception as exc:
               print(f"[{now_str}] VIX QUOTE ERROR ({VIX_SYMBOL}): {exc}")


           chain = client.get_chain(OPTION_SYMBOL, expiration)
           if not chain:
               raise RuntimeError(f"Empty chain for {OPTION_SYMBOL} {expiration}")


           try:
               daily_bars, intraday_bars = price_feed.refresh()
           except Exception as exc:
               print(f"[{now_str}] PRICE HISTORY ERROR: {exc}")
               daily_bars, intraday_bars = price_feed.daily, price_feed.intraday


           price_structure = build_price_levels(daily_bars, intraday_bars, spot)
           price_structure["smc"] = compute_smc_context(intraday_bars)
           price_structure["multi_timeframe"] = build_multi_timeframe_context(
               daily_bars, price_feed.hourly, price_feed.bars_15m, price_feed.bars_1m,
           )
           if vix_value is None:
               vix_bars = normalize_bars(price_feed.vix_daily)
               if vix_bars:
                   vix_value = vix_bars[-1]["close"]
                   vix_source = f"{VIX_SYMBOL} daily close {vix_bars[-1]['time'].date()}"


           gex = compute_gex(chain, spot, level_window=GAMMA_LEVEL_WINDOW)
           atm_straddle = _atm_straddle(chain, spot)
           now_et = datetime.now(ZoneInfo("America/New_York"))
           session_playbook = _build_session_playbook(
               spot, gex.get("net_gex"), gex.get("gamma_levels"), gex.get("flip_level"),
               price_structure, vix_value, vix_source, atm_straddle,
               daily_bars, price_feed.vix_daily, intraday_bars,
               as_of=now_et.date(), now=now_et,
           )
           today = date.today().isoformat()
           if _gex_baseline is None or _gex_baseline["date"] != today:
               _gex_baseline = {
                   "date": today,
                   "time": now_str,
                   "table": [dict(row) for row in gex.get("table", [])],
               }
           gex_change = _build_gex_change(
               gex.get("table", []), _gex_baseline["table"],
               _gex_baseline["time"], spot,
               has_comparison=_gex_baseline["time"] != now_str,
           )
           max_pain = compute_max_pain(chain)
           volume_meter = compute_volume_meter(gex)
           dex = compute_dex(chain, spot)
           flow = compute_flow(chain)
           live_flow, flow_events = flow_tracker.update(chain, spot)
           try:
               _persist_daily_data(spot, expiration, gex.get("table", []), flow_events)
           except OSError as exc:
               print(f"[{now_str}] CSV PERSISTENCE ERROR: {exc}")
           display_strikes = _nearest_strikes(chain, spot)
           live_flow = _filter_live_flow_for_spot(live_flow, display_strikes)
           heatmap = build_options_heatmap(
               chain, flow_events, spot, level_window=STRIKE_WINDOW,
           )
           dealer_supply_demand = build_dealer_supply_demand(
               gex.get("table", []), spot, level_window=STRIKE_WINDOW,
           )
           vex = compute_vex(chain)
           leaders = compute_leaderboards(gex.get("table", []), top_n=LEADERBOARD_SIZE)
           trimmed = _trim_by_spot(gex.get("table", []), spot)
           dex_trimmed = _trim_by_spot(dex.get("table", []), spot)
           vex_trimmed = _trim_by_spot(vex.get("table", []), spot)


           summary = _build_summary(spot, gex, dex, price_structure)
           summary["flow_score"] = flow.get("directional_score", 0.0)
           trade_idea = build_trade_idea(summary)


           if prev_summary is not None:
               try:
                   structural = detect_events(prev_summary, summary)
                   with _lock:
                       _append_structural_events(structural, now_str)
               except Exception as exc:
                   print(f"[{now_str}] EVENT DETECTION ERROR: {exc}")


           try:
               trade_plan = generate_trade_plan_events(summary)
               if not isinstance(trade_plan, list):
                   trade_plan = []
           except Exception as exc:
               print(f"[{now_str}] TRADE PLAN ERROR: {exc}")
               trade_plan = []


           try:
               important_levels = collect_important_levels(gex, price_structure)
               level_events = update_level_states(
                   important_levels, spot, prev_spot, _level_state_store,
               )
               level_states = build_level_state_rows(important_levels, spot, _level_state_store)
               commentary = generate_market_commentary(
                   gex,
                   price_structure,
                   spot,
                   level_events=level_events,
                   smc_context=price_structure.get("smc"),
                   multi_timeframe=price_structure.get("multi_timeframe"),
               ) + generate_level_commentary(level_states, gex.get("net_gex", 0.0))
           except Exception as exc:
               print(f"[{now_str}] LEVEL STATE ERROR: {exc}")
               level_events, level_states, commentary = [], [], []


           with _lock:
               _record_trade_idea(trade_idea, now_str)
               _append_trade_plan_events(trade_plan, now_str)
               _append_structural_events(level_events, now_str)
               _latest.update({
                   **summary,
                   "gross_call_gex": gex.get("gross_call_gex"),
                   "gross_put_gex": gex.get("gross_put_gex"),
                   "local_gamma_transition": gex.get("local_gamma_transition"),
                   "gamma_levels": gex.get("gamma_levels", {}),
                   "level_states": level_states,
                   "commentary": commentary,
                   "price_structure": {
                       "prior_day": price_structure.get("prior_day"),
                       "prior_week": price_structure.get("prior_week"),
                       "sessions": price_structure.get("sessions"),
                       "vwap": price_structure.get("vwap"),
                       "smc": price_structure.get("smc", {}),
                       "multi_timeframe": price_structure.get("multi_timeframe", {}),
                   },
                   "price_bars": _chart_bars(intraday_bars),
                   "price_session_ohlc": _session_ohlc(intraday_bars),
                   "vix": vix_value,
                   "vix_source": vix_source,
                   "atm_straddle": atm_straddle,
                   "session_playbook": session_playbook,
                   "max_pain": max_pain,
                   "table": trimmed,
                   "gex_change": gex_change,
                   "volume_meter": volume_meter,
                   "contract_statistics": _build_contract_statistics(
                       chain, live_flow.get("events", []),
                   ),
                   "leaders": leaders,
                   "live_flow": {
                       **live_flow,
                       "source": "Tradier chain volume increments; aggressor side is heuristic",
                   },
                   "options_heatmap": heatmap,
                   "dealer_supply_demand": dealer_supply_demand,
                   "net_dex": dex.get("net_dex"),
                   "dex_table": dex_trimmed,
                   "total_call_vega": vex.get("total_call_vega"),
                   "total_put_vega": vex.get("total_put_vega"),
                   "vex_table": vex_trimmed,
                   "trade_plan": trade_plan,
                   "trade_idea": trade_idea,
                   "updated_at": now_str,
                   "error": None,
               })


           net_dex = dex.get("net_dex") or 0.0
           call_vega = vex.get("total_call_vega") or 0.0
           put_vega = vex.get("total_put_vega") or 0.0
           print(
               f"[{now_str}] spot={spot:.2f} "
               f"net_gex={summary['net_gex']/1e9:.2f}B "
               f"gamma_above={summary['upper_gamma_wall']} "
               f"gamma_below={summary['lower_gamma_wall']} "
               f"flip={summary['flip_level']} "
               f"call_vol={volume_meter.get('call_volume', 0)} "
               f"put_vol={volume_meter.get('put_volume', 0)} "
               f"net_dex={net_dex/1e6:.1f}M "
               f"call_vega={call_vega/1e6:.1f}M put_vega={put_vega/1e6:.1f}M"
           )
           print(
               f"[{now_str}] TRADE IDEA: {trade_idea['action']} "
               f"entry={_format_price(trade_idea.get('entry'))} "
               f"target={_format_price(trade_idea.get('target'))} "
               f"TP2={_format_price(trade_idea.get('tp2'))} "
               f"TP3={_format_price(trade_idea.get('tp3'))} "
               f"stop={_format_price(trade_idea.get('stop_loss'))} "
               f"SMC={trade_idea['smc_bias']}"
           )
           prev_summary = summary
           prev_spot = spot


       except Exception as exc:
           print(f"[{now_str}] POLL ERROR: {exc}")
           with _lock:
               _latest["error"] = str(exc)
               _latest["updated_at"] = now_str


       time.sleep(POLL_SECONDS)




@app.route("/")
def index():
   return render_template("index.html", symbol=UNDERLYING_SYMBOL,
                                             option_symbol=OPTION_SYMBOL,
                                  expiration=_expiration(), poll_seconds=POLL_SECONDS,
                                             trade_major_shift_points=TRADE_MAJOR_SHIFT_POINTS,
                                             export_date=date.today().isoformat())




@app.route("/api/state")
def api_state():
   with _lock:
       return jsonify({
           "latest": dict(_latest),
           "events": list(reversed(_events[-50:])),
           "trade_history": list(reversed(_trade_history)),
       })


def _ai_market_context():
   with _lock:
       state = dict(_latest)
   idea = state.get("trade_idea") or {}
   flow = state.get("live_flow") or {}
   gamma_levels = state.get("gamma_levels") or {}
   spot = _safe_float(state.get("spot"))
   nearby_strikes = [
       row for row in (state.get("table") or [])
       if _safe_float(row.get("strike")) is not None
   ]
   if spot is not None:
       nearby_strikes.sort(key=lambda row: abs(float(row["strike"]) - spot))
   return {
       "symbol": UNDERLYING_SYMBOL,
       "option_symbol": OPTION_SYMBOL,
       "expiration": _expiration(),
       "updated_at": state.get("updated_at"),
       "spot": state.get("spot"),
       "gex": {
           "net": state.get("net_gex"),
           "gross_call": state.get("gross_call_gex"),
           "gross_put": state.get("gross_put_gex"),
           "flip": state.get("flip_level"),
           "upper_wall": state.get("upper_gamma_wall"),
           "lower_wall": state.get("lower_gamma_wall"),
           "call_wall_above": state.get("call_wall_above"),
           "put_wall_below": state.get("put_wall_below"),
           "levels_above": (gamma_levels.get("levels_above") or [])[:8],
           "levels_below": (gamma_levels.get("levels_below") or [])[:8],
       },
       "price_levels": (state.get("price_levels") or [])[:30],
       "price_structure": state.get("price_structure") or {},
       "recent_bars": (state.get("price_bars") or [])[-12:],
    "nearby_strikes": nearby_strikes[:24],
       "trade_idea": {
           key: idea.get(key) for key in (
               "action", "entry", "target", "tp1", "tp2", "tp3", "stop_loss",
               "confidence", "reason", "smc_bias", "timeframe_biases", "timeframe_alignment",
               "htf_bias", "entry_1m_bias", "entry_1m_confirmation", "bos",
               "liquidity_path", "active_fvg", "inverse_fvg",
           ) if key in idea
       },
       "live_flow": {
           "directional_score": flow.get("directional_score"),
           "totals": flow.get("totals") or {},
           "recent_events": [
               {key: event.get(key) for key in (
                   "time", "option_type", "strike", "side", "contracts", "last",
               ) if key in event}
               for event in (flow.get("events") or [])[-12:]
           ],
       },
       "volume_meter": state.get("volume_meter") or {},
       "commentary": (state.get("commentary") or [])[-8:],
   }


@app.route("/api/ai/providers")
def api_ai_providers():
   return jsonify({
       "openai": bool(os.environ.get("OPENAI_API_KEY")),
       "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
   })


@app.route("/api/ai/chat", methods=["POST"])
def api_ai_chat():
   payload = request.get_json(silent=True) or {}
   if not isinstance(payload, dict):
       return jsonify({"error": "Invalid chat request."}), 400
   provider = payload.get("provider")
   message = payload.get("message")
   if provider not in {"openai", "claude"}:
       return jsonify({"error": "Choose OpenAI or Claude."}), 400
   if not isinstance(message, str) or not message.strip() or len(message) > 2000:
       return jsonify({"error": "Enter a message up to 2,000 characters."}), 400

   history = payload.get("messages") or []
   if not isinstance(history, list):
       return jsonify({"error": "Invalid chat history."}), 400
   safe_history = []
   for item in history[-10:]:
       if (not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}
               or not isinstance(item.get("content"), str)):
           return jsonify({"error": "Invalid chat history."}), 400
       safe_history.append({"role": item["role"], "content": item["content"][:2000]})
   safe_history.append({"role": "user", "content": message.strip()})

   system_prompt = (
       "You are a market-structure assistant analyzing the supplied live dashboard snapshot. "
       "Give conditional decision support, not a guaranteed prediction or personalized financial advice. "
       "Start with BUY, SELL, or WAIT for the underlying direction, and state whether that means underlying "
       "price direction, not a specific options contract. Cite the actual spot, GEX regime, nearby levels, "
       "and relevant flow/structure evidence. Provide a trigger, invalidation, and next levels only when "
       "supported by the snapshot. If evidence conflicts, data is stale/missing, or no trigger is confirmed, "
       "say WAIT and explain what would confirm either side. Never invent values; distinguish observed data "
       "from inference. Keep the answer concise."
   )
   market_context = _ai_market_context()
   user_content = (
       "Current dashboard snapshot (JSON):\n"
       + json.dumps(market_context, separators=(",", ":"), default=str)
       + "\n\nUser question: " + message.strip()
   )
   safe_history[-1]["content"] = user_content

   if provider == "openai":
       api_key = os.environ.get("OPENAI_API_KEY")
       if not api_key:
           return jsonify({"error": "Set OPENAI_API_KEY on the server and restart the dashboard."}), 503
       url = "https://api.openai.com/v1/chat/completions"
       headers = {"Authorization": f"Bearer {api_key}"}
       body = {
           "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
           "messages": [{"role": "system", "content": system_prompt}, *safe_history],
           "temperature": 0.2,
           "max_tokens": 700,
       }
   else:
       api_key = os.environ.get("ANTHROPIC_API_KEY")
       if not api_key:
           return jsonify({"error": "Set ANTHROPIC_API_KEY on the server and restart the dashboard."}), 503
       url = "https://api.anthropic.com/v1/messages"
       headers = {
           "x-api-key": api_key,
           "anthropic-version": "2023-06-01",
       }
       body = {
           "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
           "system": system_prompt,
           "messages": safe_history,
           "max_tokens": 700,
       }

   try:
       response = requests.post(url, headers=headers, json=body, timeout=30)
       response.raise_for_status()
       result = response.json()
   except requests.RequestException:
       app.logger.exception("AI provider request failed")
       return jsonify({"error": "The AI provider request failed. Check the server log and provider configuration."}), 502
   except ValueError:
       return jsonify({"error": "The AI provider returned an unreadable response."}), 502

   if provider == "openai":
       answer = (((result.get("choices") or [{}])[0].get("message") or {}).get("content"))
   else:
       answer = "\n".join(
           block.get("text", "") for block in (result.get("content") or [])
           if block.get("type") == "text"
       )
   return jsonify({"answer": answer, "updated_at": market_context.get("updated_at")}) if answer else (jsonify({"error": "The AI provider returned an empty answer."}), 502)


@app.route("/exports/<dataset>/<export_date>.csv")
def download_daily_csv(dataset, export_date):
   if dataset not in {"gex", "options_flow"}:
       abort(404)
   try:
       parsed_date = date.fromisoformat(export_date)
   except ValueError:
       abort(400)
   if parsed_date.isoformat() != export_date:
       abort(400)
   path = DATA_DIR / f"{export_date}_{dataset}.csv"
   if not path.is_file():
       abort(404)
   return send_file(path, mimetype="text/csv", as_attachment=True, download_name=path.name)




def main():
   if not TRADIER_TOKEN:
       raise SystemExit("Set the TRADIER_TOKEN environment variable before starting.")
   threading.Thread(target=_poll_loop, daemon=True, name="tradier-poller").start()
   app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)




if __name__ == "__main__":
   main()
