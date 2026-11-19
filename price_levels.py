"""
Price-structure levels: prior day/week extremes, session ranges and VWAP.


All timestamps are interpreted in US/Eastern, matching Tradier's market data.
Sessions that have no bars simply return None instead of being fabricated.
"""


from __future__ import annotations


from datetime import date, datetime, timedelta
from math import sqrt
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo


# name -> (start_minute_of_day, end_minute_of_day). A start >= end wraps midnight.
DEFAULT_SESSIONS: Dict[str, Tuple[int, int]] = {
   "ASIA": (18 * 60, 3 * 60),
   "LONDON": (3 * 60, 8 * 60),
   "NYAM": (9 * 60 + 30, 12 * 60),
   "NYPM": (12 * 60, 16 * 60),
}




def _to_float(value: Any) -> Optional[float]:
   if value is None or value == "":
       return None
   try:
       result = float(value)
   except (TypeError, ValueError):
       return None
   return result




def _parse_time(value: Any) -> Optional[datetime]:
   if isinstance(value, datetime):
       return value
   if not isinstance(value, str) or not value:
       return None
   try:
       return datetime.fromisoformat(value.replace("Z", "").strip())
   except ValueError:
       return None




def normalize_bars(bars: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
   """Convert raw Tradier bars into a sorted list of OHLCV dicts with datetimes."""
   result = []
   for bar in bars or []:
       stamp = _parse_time(bar.get("time") or bar.get("date"))
       close = _to_float(bar.get("close"))
       if close is None:
           close = _to_float(bar.get("price"))
       if stamp is None or close is None:
           continue


       high = _to_float(bar.get("high"))
       low = _to_float(bar.get("low"))
       open_ = _to_float(bar.get("open"))
       result.append({
           "time": stamp,
           "open": open_ if open_ is not None else close,
           "high": high if high is not None else close,
           "low": low if low is not None else close,
           "close": close,
           "volume": _to_float(bar.get("volume")) or 0.0,
           "vwap": _to_float(bar.get("vwap")),
       })


   result.sort(key=lambda b: b["time"])
   return result




def _range(bars: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
   if not bars:
       return None
   return {
       "high": max(b["high"] for b in bars),
       "low": min(b["low"] for b in bars),
       "open": bars[0]["open"],
       "close": bars[-1]["close"],
   }




def session_day(stamp: datetime, start_minute: int, end_minute: int) -> datetime:
   """The calendar day a bar belongs to for a possibly overnight session."""
   if start_minute < end_minute:
       return stamp.replace(hour=0, minute=0, second=0, microsecond=0)
   # Overnight session: bars after the start belong to the next calendar day.
   day = stamp.replace(hour=0, minute=0, second=0, microsecond=0)
   return day + timedelta(days=1) if stamp.hour * 60 + stamp.minute >= start_minute else day




def compute_session_levels(
   bars: Iterable[Dict[str, Any]],
   sessions: Dict[str, Tuple[int, int]] = DEFAULT_SESSIONS,
) -> Dict[str, Optional[Dict[str, float]]]:
   """High/low/open/close for each named session of the most recent session day."""
   normalized = normalize_bars(bars)
   result: Dict[str, Optional[Dict[str, float]]] = {}


   for name, (start, end) in sessions.items():
       in_session = []
       for bar in normalized:
           minute = bar["time"].hour * 60 + bar["time"].minute
           inside = start <= minute < end if start < end else (minute >= start or minute < end)
           if inside:
               in_session.append(bar)


       if not in_session:
           result[name] = None
           continue


       latest_day = max(session_day(b["time"], start, end) for b in in_session)
       today = [b for b in in_session if session_day(b["time"], start, end) == latest_day]
       result[name] = _range(today)


   return result




def compute_prior_day(
    daily_bars: Iterable[Dict[str, Any]],
    as_of: Optional[date] = None,
) -> Optional[Dict[str, float]]:
   """Prior completed daily bar (PDH/PDL/PDC)."""
   bars = normalize_bars(daily_bars)
   current_date = as_of or datetime.now(ZoneInfo("America/New_York")).date()
   completed = [bar for bar in bars if bar["time"].date() < current_date]
   if not completed:
       return None
   return _range([completed[-1]])




def compute_prior_week(daily_bars: Iterable[Dict[str, Any]]) -> Optional[Dict[str, float]]:
   """Prior completed ISO week (PWH/PWL)."""
   bars = normalize_bars(daily_bars)
   if not bars:
       return None


   weeks: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
   for bar in bars:
       weeks.setdefault(bar["time"].isocalendar()[:2], []).append(bar)


   ordered = sorted(weeks)
   if len(ordered) < 2:
       return None
   return _range(weeks[ordered[-2]])




def compute_vwap(bars: Iterable[Dict[str, Any]]) -> Optional[float]:
   """Session VWAP over the most recent calendar day of intraday bars."""
   normalized = normalize_bars(bars)
   if not normalized:
       return None


   last_day = normalized[-1]["time"].date()
   today = [b for b in normalized if b["time"].date() == last_day]


   volume = sum(b["volume"] for b in today)
   if volume <= 0:
       closes = [b["close"] for b in today]
       return sum(closes) / len(closes) if closes else None


   typical = sum(((b["high"] + b["low"] + b["close"]) / 3) * b["volume"] for b in today)
   return typical / volume


def compute_expected_move_calibration(
   daily_bars: Iterable[Dict[str, Any]],
   volatility_bars: Iterable[Dict[str, Any]],
   current_move: Optional[float],
   as_of: date,
   lookback: int = 60,
) -> Dict[str, Any]:
   """Compare a VIX-implied move with prior matched SPX/VIX sessions."""
   spx_by_date = {bar["time"].date(): bar for bar in normalize_bars(daily_bars)}
   vix_by_date = {bar["time"].date(): bar for bar in normalize_bars(volatility_bars)}
   dates = sorted(day for day in set(spx_by_date) & set(vix_by_date) if day < as_of)
   observations = []

   for index in range(1, len(dates)):
       previous_date, session_date = dates[index - 1], dates[index]
       previous_spx = spx_by_date[previous_date]
       previous_vix = vix_by_date[previous_date]["close"]
       session = spx_by_date[session_date]
       expected_move = previous_spx["close"] * previous_vix / (100 * sqrt(252))
       if expected_move <= 0:
           continue
       observations.append({
           "move": expected_move,
           "close_inside": abs(session["close"] - previous_spx["close"]) <= expected_move,
           "range_inside": (
               session["high"] <= previous_spx["close"] + expected_move
               and session["low"] >= previous_spx["close"] - expected_move
           ),
       })

   observations = observations[-lookback:]
   count = len(observations)
   if not count:
       return {
           "count": 0,
           "lookback": lookback,
           "percentile": None,
           "close_inside_pct": None,
           "range_inside_pct": None,
       }

   return {
       "count": count,
       "lookback": lookback,
       "percentile": (
           sum(item["move"] <= current_move for item in observations) * 100 / count
           if current_move is not None else None
       ),
       "close_inside_pct": sum(item["close_inside"] for item in observations) * 100 / count,
       "range_inside_pct": sum(item["range_inside"] for item in observations) * 100 / count,
   }


def _detect_fvg_zones(recent: List[Dict[str, Any]]) -> Dict[str, Any]:
   fvgs = []
   for index in range(2, len(recent)):
       first, third = recent[index - 2], recent[index]
       if third["low"] > first["high"]:
           fvgs.append({"direction": "BULLISH", "low": first["high"], "high": third["low"], "index": index})
       elif third["high"] < first["low"]:
           fvgs.append({"direction": "BEARISH", "low": third["high"], "high": first["low"], "index": index})
   latest = recent[-1] if recent else {}
   inverse_fvgs = []
   active_fvgs = []
   for fvg in fvgs[-10:]:
       violated = (
           latest.get("close", 0) < fvg["low"] if fvg["direction"] == "BULLISH"
           else latest.get("close", 0) > fvg["high"]
       )
       zone = {key: value for key, value in fvg.items() if key != "index"}
       if violated:
           zone["direction"] = "BEARISH" if fvg["direction"] == "BULLISH" else "BULLISH"
           zone["type"] = "INVERSE_FVG"
           inverse_fvgs.append(zone)
       else:
           zone["type"] = "FVG"
           active_fvgs.append(zone)
   return {
       "fvgs": active_fvgs,
       "active_fvg": active_fvgs[-1] if active_fvgs else None,
       "inverse_fvgs": inverse_fvgs,
   }


def _price_action_context(
   recent: List[Dict[str, Any]],
   bias: str,
   bos: Optional[str],
   liquidity_sweep: Optional[str],
   swing_highs: List[Tuple[int, float]],
   swing_lows: List[Tuple[int, float]],
) -> Dict[str, Any]:
   latest = recent[-1] if recent else {}
   ranges = [max(bar["high"] - bar["low"], 0.0) for bar in recent[-21:]]
   average_range = mean(ranges[:-1]) if len(ranges) > 1 else 0.0
   latest_range = ranges[-1] if ranges else 0.0
   body = abs(latest.get("close", 0.0) - latest.get("open", 0.0))
   body_ratio = body / latest_range if latest_range else 0.0
   close_location = (
       (latest["close"] - latest["low"]) / latest_range
       if latest_range else 0.5
   )
   displacement = latest_range >= average_range * 1.25 and body_ratio >= 0.55

   bullish_progression = (
       len(swing_highs) >= 2 and len(swing_lows) >= 2
       and swing_highs[-1][1] > swing_highs[-2][1]
       and swing_lows[-1][1] > swing_lows[-2][1]
   )
   bearish_progression = (
       len(swing_highs) >= 2 and len(swing_lows) >= 2
       and swing_highs[-1][1] < swing_highs[-2][1]
       and swing_lows[-1][1] < swing_lows[-2][1]
   )

   score = 25
   strength_signals = []
   weakness_signals = []
   if bias == "BULLISH":
       score += 25
       strength_signals.append("higher highs and higher lows")
   elif bias == "BEARISH":
       score += 25
       strength_signals.append("lower highs and lower lows")
   if bullish_progression or bearish_progression:
       score += 10
   if bos == bias and bias in {"BULLISH", "BEARISH"}:
       score += 20
       strength_signals.append(f"{bias.lower()} break of structure")
   elif bos:
       score -= 25
       weakness_signals.append(f"{bos.lower()} break of structure against trend")
   if liquidity_sweep == bias and bias in {"BULLISH", "BEARISH"}:
       score += 10
       strength_signals.append(f"{bias.lower()} liquidity sweep")
   elif liquidity_sweep:
       score -= 15
       weakness_signals.append(f"{liquidity_sweep.lower()} liquidity sweep against trend")
   if displacement and bias in {"BULLISH", "BEARISH"}:
       candle_direction = "BULLISH" if latest["close"] > latest["open"] else "BEARISH"
       if candle_direction == bias and (
           (bias == "BULLISH" and close_location >= 0.7)
           or (bias == "BEARISH" and close_location <= 0.3)
       ):
           score += 15
           strength_signals.append("directional displacement")
       elif candle_direction != bias:
           score -= 15
           weakness_signals.append("strong counter-trend candle")

   last_swing_high = swing_highs[-1][1] if swing_highs else None
   last_swing_low = swing_lows[-1][1] if swing_lows else None
   if bias == "BULLISH" and last_swing_low is not None:
       weakness_level = last_swing_low
       strength_level = last_swing_high
       if latest and latest["close"] < weakness_level:
           score -= 30
           weakness_signals.append("close below protected swing low")
   elif bias == "BEARISH" and last_swing_high is not None:
       weakness_level = last_swing_high
       strength_level = last_swing_low
       if latest and latest["close"] > weakness_level:
           score -= 30
           weakness_signals.append("close above protected swing high")
   else:
       weakness_level = None
       strength_level = None

   score = max(0, min(100, score))
   if bias == "NEUTRAL" and not bos:
       strength = "NEUTRAL"
   elif score >= 70:
       strength = "STRONG"
   elif score >= 45:
       strength = "DEVELOPING"
   else:
       strength = "WEAK"

   recent_range = max((bar["high"] for bar in recent[-8:]), default=0.0) - min(
       (bar["low"] for bar in recent[-8:]), default=0.0,
   )
   compression_ratio = recent_range / (average_range * min(8, len(recent))) if average_range and recent else None
   if liquidity_sweep:
       amd_phase = "MANIPULATION"
       amd_direction = liquidity_sweep
       amd_evidence = f"{liquidity_sweep.lower()} sweep of a recent swing, then close back through the level"
   elif bos and displacement:
       amd_phase = "DISTRIBUTION"
       amd_direction = bos
       amd_evidence = f"{bos.lower()} structure break with displacement"
   elif compression_ratio is not None and compression_ratio <= 0.65 and not bos:
       amd_phase = "ACCUMULATION"
       amd_direction = "RANGE"
       amd_evidence = "compressed recent range without a confirmed structure break"
   else:
       amd_phase = "UNCONFIRMED"
       amd_direction = None
       amd_evidence = "no clear compression, sweep, or displacement break"

   return {
       "trend_strength": strength,
       "trend_strength_score": score,
       "strength_signals": strength_signals,
       "weakness_signals": weakness_signals,
       "strength_level": strength_level,
    "strength_condition": "ABOVE" if bias == "BULLISH" else "BELOW" if bias == "BEARISH" else None,
       "weakness_level": weakness_level,
    "weakness_condition": "BELOW" if bias == "BULLISH" else "ABOVE" if bias == "BEARISH" else None,
       "amd_phase": amd_phase,
       "amd_direction": amd_direction,
       "amd_evidence": amd_evidence,
       "range_compression_ratio": round(compression_ratio, 3) if compression_ratio is not None else None,
   }


def compute_smc_context(
   bars: Iterable[Dict[str, Any]],
   lookback: int = 80,
   swing_strength: int = 2,
) -> Dict[str, Any]:
   """Return deterministic SMC structure, FVG, and liquidity context."""
   recent = normalize_bars(bars)[-lookback:]
   empty = {
       "bias": "NEUTRAL", "bos": None, "liquidity_sweep": None,
       "swing_high": None, "swing_low": None, "order_block": None,
       "fvgs": [], "active_fvg": None, "inverse_fvgs": [],
       "internal_liquidity": [], "external_liquidity": {},
       "liquidity_path": None, "trend_strength": "NEUTRAL",
       "trend_strength_score": 25, "strength_signals": [],
       "weakness_signals": [], "strength_level": None,
       "weakness_level": None, "amd_phase": "UNCONFIRMED",
       "amd_direction": None, "amd_evidence": "insufficient structure data",
       "range_compression_ratio": None,
   }
   gap_context = _detect_fvg_zones(recent)
   if len(recent) < swing_strength * 2 + 3:
       return {
           **empty,
           **gap_context,
           **_price_action_context(recent, "NEUTRAL", None, None, [], []),
       }

   swing_highs = []
   swing_lows = []
   for index in range(swing_strength, len(recent) - swing_strength):
       bar = recent[index]
       left = recent[index - swing_strength:index]
       right = recent[index + 1:index + swing_strength + 1]
       if bar["high"] > max(x["high"] for x in left + right):
           swing_highs.append((index, bar["high"]))
       if bar["low"] < min(x["low"] for x in left + right):
           swing_lows.append((index, bar["low"]))

   if not swing_highs or not swing_lows:
       return {
           **empty,
           **gap_context,
           **_price_action_context(recent, "NEUTRAL", None, None, swing_highs, swing_lows),
           "external_liquidity": {
               "buy_side": {"label": "EXTERNAL_BSL", "price": max(x["high"] for x in recent)},
               "sell_side": {"label": "EXTERNAL_SSL", "price": min(x["low"] for x in recent)},
           },
       }

   last_high = swing_highs[-1]
   last_low = swing_lows[-1]
   bias = "NEUTRAL"
   if len(swing_highs) >= 2 and len(swing_lows) >= 2:
       higher_structure = (
           swing_highs[-1][1] > swing_highs[-2][1]
           and swing_lows[-1][1] > swing_lows[-2][1]
       )
       lower_structure = (
           swing_highs[-1][1] < swing_highs[-2][1]
           and swing_lows[-1][1] < swing_lows[-2][1]
       )
       bias = "BULLISH" if higher_structure else "BEARISH" if lower_structure else bias

   latest = recent[-1]
   bos = None
   if latest["close"] > last_high[1]:
       bos = "BULLISH"
   elif latest["close"] < last_low[1]:
       bos = "BEARISH"

   liquidity_sweep = None
   if latest["low"] < last_low[1] and latest["close"] > last_low[1]:
       liquidity_sweep = "BULLISH"
   elif latest["high"] > last_high[1] and latest["close"] < last_high[1]:
       liquidity_sweep = "BEARISH"

   order_block = None
   if bos:
       for bar in reversed(recent[:-1]):
           opposite = bar["close"] < bar["open"] if bos == "BULLISH" else bar["close"] > bar["open"]
           if opposite:
               order_block = {"direction": bos, "high": bar["high"], "low": bar["low"]}
               break

   external_high = max(x["high"] for x in recent)
   external_low = min(x["low"] for x in recent)
   internal = [
       {"side": "BUY_SIDE", "price": price, "type": "INTERNAL"}
       for _, price in swing_highs[:-1]
       if price < external_high
   ] + [
       {"side": "SELL_SIDE", "price": price, "type": "INTERNAL"}
       for _, price in swing_lows[:-1]
       if price > external_low
   ]
   external = {
       "buy_side": {"label": "EXTERNAL_BSL", "price": external_high},
       "sell_side": {"label": "EXTERNAL_SSL", "price": external_low},
   }
   draw = "EXTERNAL_BSL" if bias == "BULLISH" else "EXTERNAL_SSL" if bias == "BEARISH" else None
   price_action = _price_action_context(
       recent, bias, bos, liquidity_sweep, swing_highs, swing_lows,
   )

   return {
       "bias": bias, "bos": bos, "liquidity_sweep": liquidity_sweep,
       "swing_high": last_high[1], "swing_low": last_low[1],
    "order_block": order_block, **gap_context,
    "internal_liquidity": internal[-6:],
    "external_liquidity": external, "liquidity_path": draw,
    **price_action,
   }


def _resample_bars(bars: Iterable[Dict[str, Any]], minutes: int) -> List[Dict[str, Any]]:
   """Aggregate normalized bars into fixed-minute OHLCV bars."""
   normalized = normalize_bars(bars)
   if minutes <= 1:
       return normalized
   groups: Dict[datetime, List[Dict[str, Any]]] = {}
   for bar in normalized:
       total_minutes = bar["time"].hour * 60 + bar["time"].minute
       bucket = total_minutes // minutes * minutes
       stamp = bar["time"].replace(
           hour=bucket // 60, minute=bucket % 60, second=0, microsecond=0,
       )
       groups.setdefault(stamp, []).append(bar)
   result = []
   for stamp, group in sorted(groups.items()):
       result.append({
           "time": stamp,
           "open": group[0]["open"],
           "high": max(x["high"] for x in group),
           "low": min(x["low"] for x in group),
           "close": group[-1]["close"],
           "volume": sum(x["volume"] for x in group),
       })
   return result


def _standard_deviation_context(bars: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
   closes = [bar["close"] for bar in normalize_bars(bars)]
   if not closes:
       return {"mean": None, "stddev": None, "upper_1": None, "lower_1": None,
               "upper_2": None, "lower_2": None}
   center = mean(closes)
   deviation = pstdev(closes) if len(closes) > 1 else 0.0
   return {
       "mean": center,
       "stddev": deviation,
       "upper_1": center + deviation,
       "lower_1": center - deviation,
       "upper_2": center + 2 * deviation,
       "lower_2": center - 2 * deviation,
   }


def _rejection_block(bars: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
   recent = normalize_bars(bars)
   if not recent:
       return None
   bar = recent[-1]
   body = max(abs(bar["close"] - bar["open"]), 0.01)
   upper_wick = bar["high"] - max(bar["open"], bar["close"])
   lower_wick = min(bar["open"], bar["close"]) - bar["low"]
   if lower_wick >= body * 1.5 and bar["close"] >= bar["open"]:
       return {"direction": "BULLISH", "low": bar["low"], "high": min(bar["open"], bar["close"])}
   if upper_wick >= body * 1.5 and bar["close"] <= bar["open"]:
       return {"direction": "BEARISH", "low": max(bar["open"], bar["close"]), "high": bar["high"]}
   return None


def compute_timeframe_context(
   bars: Iterable[Dict[str, Any]],
   label: str,
   lookback: int = 80,
) -> Dict[str, Any]:
   """Combine SMC zones, rejection blocks, and standard deviations for one timeframe."""
   normalized = normalize_bars(bars)[-lookback:]
   smc = compute_smc_context(normalized, lookback=lookback)
   return {
       "timeframe": label,
       "bias": smc.get("bias", "NEUTRAL"),
       "bos": smc.get("bos"),
       "liquidity_sweep": smc.get("liquidity_sweep"),
    "trend_strength": smc.get("trend_strength", "NEUTRAL"),
    "trend_strength_score": smc.get("trend_strength_score", 0),
    "strength_level": smc.get("strength_level"),
    "weakness_level": smc.get("weakness_level"),
    "weakness_signals": smc.get("weakness_signals", []),
    "amd_phase": smc.get("amd_phase", "UNCONFIRMED"),
    "amd_direction": smc.get("amd_direction"),
    "amd_evidence": smc.get("amd_evidence"),
       "order_block": smc.get("order_block"),
       "active_fvg": smc.get("active_fvg"),
       "inverse_fvgs": smc.get("inverse_fvgs", []),
       "rejection_block": _rejection_block(normalized),
       "stddev": _standard_deviation_context(normalized),
       "close": normalized[-1]["close"] if normalized else None,
   }


def build_multi_timeframe_context(
   daily_bars: Iterable[Dict[str, Any]],
   hourly_bars: Iterable[Dict[str, Any]],
   bars_15m: Iterable[Dict[str, Any]],
   bars_1m: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
   """Build Weekly/Day/4H/1H/15M bias and 1M execution context."""
   daily = normalize_bars(daily_bars)
   hourly = normalize_bars(hourly_bars)
   contexts = {
       "WEEKLY": compute_timeframe_context(_resample_bars(daily, 5), "WEEKLY"),
       "DAY": compute_timeframe_context(daily, "DAY"),
       "4H": compute_timeframe_context(_resample_bars(hourly, 240), "4H"),
       "1H": compute_timeframe_context(hourly, "1H"),
       "15M": compute_timeframe_context(bars_15m, "15M"),
   }
   entry = compute_timeframe_context(bars_1m, "1M", lookback=120)
   bullish = sum(context["bias"] == "BULLISH" for context in contexts.values())
   bearish = sum(context["bias"] == "BEARISH" for context in contexts.values())
   htf_bias = "BULLISH" if bullish > bearish else "BEARISH" if bearish > bullish else "NEUTRAL"
   entry_bias = entry["bias"]
   return {
       "timeframes": contexts,
       "htf_bias": htf_bias,
       "htf_bullish_count": bullish,
       "htf_bearish_count": bearish,
       "entry_1m": entry,
       "entry_alignment": (
           "ALIGNED" if entry_bias == htf_bias and htf_bias != "NEUTRAL"
           else "WAIT" if htf_bias != "NEUTRAL" else "UNDEFINED"
       ),
   }




def build_price_levels(
    daily_bars: Iterable[Dict[str, Any]],
    intraday_bars: Iterable[Dict[str, Any]],
    spot: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble price-structure levels consumed by the dashboard."""
    prior_day = compute_prior_day(daily_bars)
    prior_week = compute_prior_week(daily_bars)
    sessions = compute_session_levels(intraday_bars)
    vwap = compute_vwap(intraday_bars)
    levels: List[Dict[str, Any]] = []

    def add(name: str, price: Optional[float], kind: str) -> None:
        if price is None:
            return
        entry: Dict[str, Any] = {"name": name, "price": price, "kind": kind}
        if spot is not None:
            entry["distance"] = price - spot
            entry["side"] = "above" if price > spot else "below"
        levels.append(entry)

    if prior_day:
        add("PDH", prior_day["high"], "daily")
        add("PDL", prior_day["low"], "daily")
        add("PDC", prior_day["close"], "daily")
    if prior_week:
        add("PWH", prior_week["high"], "weekly")
        add("PWL", prior_week["low"], "weekly")
    for name, rng in sessions.items():
        if rng:
            add(f"{name} HIGH", rng["high"], "session")
            add(f"{name} LOW", rng["low"], "session")
    add("VWAP", vwap, "vwap")

    return {
        "prior_day": prior_day,
        "prior_week": prior_week,
        "sessions": sessions,
        "vwap": vwap,
        "levels": sorted(levels, key=lambda x: x["price"], reverse=True),
    }
