import os
import time
import threading
from collections import deque
from datetime import date, datetime, timedelta


from flask import Flask, jsonify, render_template
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
PRICE_HISTORY_DAYS = int(os.environ.get("PRICE_HISTORY_DAYS", "20"))
PRICE_REFRESH_SECONDS = max(30, int(os.environ.get("PRICE_REFRESH_SECONDS", "60")))
PORT = int(os.environ.get("PORT", "5000"))


app = Flask(__name__)
_lock = threading.Lock()
_latest = {
   "spot": None, "net_gex": None, "gross_call_gex": None, "gross_put_gex": None,
   "call_wall": None, "put_wall": None, "flip_level": None,
   "upper_gamma_wall": None, "lower_gamma_wall": None,
   "call_wall_above": None, "put_wall_below": None,
   "gamma_levels": {}, "price_levels": [], "price_structure": {}, "price_bars": [],
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
   "updated_at": None, "error": None,
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
                   "cvd": {"series": [], "strike_table": []}}, []

       flow = compute_live_flow(self.previous, chain)
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
           for field in ("action", "smc_bias", "bos", "liquidity_path", "active_fvg", "inverse_fvg")
       )
       level_shift = any(
           price_shift(field) >= TRADE_MAJOR_SHIFT_POINTS
           for field in ("target", "stop_loss")
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




class _PriceFeed:
   """Fetches daily and intraday bars, refreshed on its own slower cadence."""


   def __init__(self, client, symbol):
       self.client = client
       self.symbol = symbol
       self.daily = []
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


           gex = compute_gex(chain, spot, level_window=GAMMA_LEVEL_WINDOW)
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
                   gex, price_structure, spot, level_events=level_events,
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
                   "max_pain": max_pain,
                   "table": trimmed,
                   "gex_change": gex_change,
                   "volume_meter": volume_meter,
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
               f"entry={trade_idea['entry']:.2f} target={trade_idea['target']:.2f} "
               f"stop={trade_idea['stop_loss']:.2f} "
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
                                  trade_major_shift_points=TRADE_MAJOR_SHIFT_POINTS)




@app.route("/api/state")
def api_state():
   with _lock:
       return jsonify({
           "latest": dict(_latest),
           "events": list(reversed(_events[-50:])),
           "trade_history": list(reversed(_trade_history)),
       })




def main():
   if not TRADIER_TOKEN:
       raise SystemExit("Set the TRADIER_TOKEN environment variable before starting.")
   threading.Thread(target=_poll_loop, daemon=True, name="tradier-poller").start()
   app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)




if __name__ == "__main__":
   main()
