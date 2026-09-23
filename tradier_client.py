"""
Minimal Tradier API client.


Docs: https://documentation.tradier.com/brokerage-api


Needs:
 - A Tradier account (paper/sandbox works for testing endpoints, but greeks
   on the sandbox chain endpoint are frequently null/delayed - use a live
   market-data-enabled account for real gamma values).
 - An access token, passed as TRADIER_TOKEN env var or the `token=` kwarg.
"""


import requests




class TradierClient:
   def __init__(self, token=None, sandbox=False):
       if not token:
           raise ValueError("A Tradier API token is required (TRADIER_TOKEN env var).")
       self.token = token
       self.base_url = "https://sandbox.tradier.com/v1" if sandbox else "https://api.tradier.com/v1"
       self.session = requests.Session()
       self.session.headers.update({
           "Authorization": f"Bearer {self.token}",
           "Accept": "application/json",
       })


   def get_quote(self, symbol):
       """Returns a single quote dict for `symbol`, or None if not found."""
       r = self.session.get(f"{self.base_url}/markets/quotes", params={"symbols": symbol, "greeks": "false"})
       r.raise_for_status()
       quotes = (r.json().get("quotes") or {}).get("quote")
       if quotes is None:
           return None
       if isinstance(quotes, list):
           return quotes[0]
       return quotes


   def get_expirations(self, symbol):
       """Returns a list of available expiration date strings ('YYYY-MM-DD') for `symbol`."""
       r = self.session.get(
           f"{self.base_url}/markets/options/expirations",
           params={"symbol": symbol, "includeAllRoots": "true"},
       )
       r.raise_for_status()
       exp = (r.json().get("expirations") or {}).get("date")
       if exp is None:
           return []
       return [exp] if isinstance(exp, str) else exp


   def get_chain(self, symbol, expiration):
       """Returns the full option chain (list of dicts, each with strike,
       option_type, open_interest, greeks{gamma,...}) for one expiration."""
       r = self.session.get(
           f"{self.base_url}/markets/options/chains",
           params={"symbol": symbol, "expiration": expiration, "greeks": "true"},
       )
       r.raise_for_status()
       options = r.json().get("options") or {}
       opts = options.get("option")
       if opts is None:
           return []
       return [opts] if isinstance(opts, dict) else opts


   def get_history(self, symbol, interval="daily", start=None, end=None):
       """Returns daily/weekly/monthly OHLCV bars (list of dicts with date, open,
       high, low, close, volume) for `symbol`."""
       params = {"symbol": symbol, "interval": interval}
       if start:
           params["start"] = start
       if end:
           params["end"] = end
       r = self.session.get(f"{self.base_url}/markets/history", params=params)
       r.raise_for_status()
       history = r.json().get("history") or {}
       days = history.get("day")
       if days is None:
           return []
       return [days] if isinstance(days, dict) else days


   def get_timesales(self, symbol, interval="5min", start=None, end=None):
       """Returns intraday bars (list of dicts with time, open, high, low, close,
       volume, vwap). Timestamps are US/Eastern."""
       params = {"symbol": symbol, "interval": interval, "session_filter": "all"}
       if start:
           params["start"] = start
       if end:
           params["end"] = end
       r = self.session.get(f"{self.base_url}/markets/timesales", params=params)
       r.raise_for_status()
       series = r.json().get("series") or {}
       data = series.get("data")
       if data is None:
           return []
       return [data] if isinstance(data, dict) else data
