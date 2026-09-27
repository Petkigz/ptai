"""
Base rates that are actually frequencies, not constants.

The base-rate model shipped with hardcoded category numbers (`politics 0.52`,
`sports 0.50`, ...) and an empty `historical_data` list nothing ever filled. On
the operator's 2026-09-27 log every market printed

    Base-rate model has no historical data loaded

so one of the five independent forecasting components contributed nothing at all
- and the honesty of the warning is exactly why it could not silently pretend
otherwise. The fix is not a better constant. It is data.

Where the data comes from
-------------------------
The venue's own resolved markets. Polymarket's Gamma API serves closed markets
with `outcomePrices` set to the settlement marks (`["1","0"]` means YES resolved,
`["0","1"]` means NO), so a few hundred closed markets give a real frequency:
"How often does a market in this category resolve YES?"

What it is and is not
---------------------
It is a PRIOR, over the population of markets that venue listed. It is not the
world's base rate and it is not a forecast about a particular question: a
category where 38% of markets resolved YES says nothing about whether *this*
market will. That is why the model built on it gets a small, sample-sized
weight, reports the count it is built from, and never overrides the market's own
price. What it can do is disagree with a price that has drifted far from the
history of its category - which is precisely the evidence the stack was missing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# Below this many resolved markets a category has no usable frequency, and the
# model keeps saying "no data" rather than publishing a number built from five.
MIN_SAMPLE = 30
STATE_KEY = "intelligence.base_rates"


def _settled_yes(row: Dict[str, Any]) -> Optional[bool]:
    """
    Did this closed market resolve YES?

    Reads the settlement marks Gamma publishes for closed markets. Anything
    unreadable returns None - an unreadable market is not a NO.
    """
    prices = row.get("outcomePrices")
    if isinstance(prices, str):
        try:
            import json
            prices = json.loads(prices)
        except (TypeError, ValueError):
            return None
    if not isinstance(prices, list) or not prices:
        return None
    try:
        first = float(prices[0])
    except (TypeError, ValueError):
        return None
    if first >= 0.999:
        return True
    if first <= 0.001:
        return False
    return None


def _category_of(row: Dict[str, Any]) -> str:
    """
    The category a market belongs to, from what the venue says about it.

    Deliberately shallow: `category`, then the first event tag, then "default".
    Guessing a category from keywords in the question is how a market ends up in
    a bucket whose frequency has nothing to do with it.
    """
    for key in ("category", "groupSlug"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    events = row.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            for tag in (event.get("tags") or []):
                if isinstance(tag, dict) and tag.get("label"):
                    return str(tag["label"]).strip().lower()
                if isinstance(tag, str) and tag.strip():
                    return tag.strip().lower()
            if event.get("category"):
                return str(event["category"]).strip().lower()
    return "default"


class BaseRateBook:
    """
    Per-category YES frequencies built from resolved markets.

    Lives in the same key-value store as everything else (`PTAI_DB`), so it
    follows the database the agent is actually using.
    """

    def __init__(self, storage: Any = None, min_sample: int = MIN_SAMPLE):
        self.storage = storage
        self.min_sample = min_sample
        self.dataset: Dict[str, Any] = {}
        if storage is not None:
            self.dataset = self.load()

    # -- persistence --------------------------------------------------------
    def load(self) -> Dict[str, Any]:
        if self.storage is None:
            return {}
        try:
            raw = self.storage.get_state(STATE_KEY)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not read the base-rate dataset: "
                         f"{type(e).__name__}: {e}")
            return {}
        if not raw:
            return {}
        try:
            import json
            payload = json.loads(raw)
        except (TypeError, ValueError) as e:
            logger.error(f"Base-rate dataset is unreadable ({e}); treating it as "
                         f"empty rather than as data")
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self) -> bool:
        if self.storage is None:
            return False
        try:
            import json
            self.storage.set_state(STATE_KEY, json.dumps(self.dataset))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not store the base-rate dataset: "
                         f"{type(e).__name__}: {e}")
            return False

    # -- building -----------------------------------------------------------
    def build_from_markets(self, rows: List[Dict[str, Any]],
                           source: str = "venue closed markets") -> Dict[str, Any]:
        """
        Count resolved markets per category. Returns the dataset.

        Counted, not inferred: a row whose settlement cannot be read is skipped
        and counted as skipped, so the sample size in the dataset is the number
        of markets that actually resolved one way or the other.
        """
        categories: Dict[str, Dict[str, Any]] = {}
        skipped = 0
        for row in rows or []:
            if not isinstance(row, dict):
                skipped += 1
                continue
            yes = _settled_yes(row)
            if yes is None:
                skipped += 1
                continue
            category = _category_of(row)
            bucket = categories.setdefault(category, {"n": 0, "yes": 0,
                                                      "prices": []})
            bucket["n"] += 1
            bucket["yes"] += 1 if yes else 0
            # The mean of what these markets LAST TRADED AT, for the comparison
            # the model needs: if the category resolves YES 38% of the time but
            # trades at 45 on average, the venue's own population says its prices
            # run hot there. `outcomePrices` is deliberately not used for this -
            # on a closed market it is the settlement mark (1 or 0), not a price.
            last = row.get("lastTradePrice")
            try:
                if last is not None:
                    bucket["prices"].append(float(last))
            except (TypeError, ValueError):
                pass

        stats: Dict[str, Dict[str, Any]] = {}
        for category, bucket in categories.items():
            n = int(bucket["n"])
            entry: Dict[str, Any] = {
                "n": n,
                "yes": int(bucket["yes"]),
                "rate": round(int(bucket["yes"]) / n, 4) if n else None,
                "usable": n >= self.min_sample,
            }
            if bucket["prices"]:
                entry["mean_last_price"] = round(
                    sum(bucket["prices"]) / len(bucket["prices"]), 4)
            stats[category] = entry

        self.dataset = {
            "source": source,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "markets_read": len(rows or []),
            "markets_skipped": skipped,
            "min_sample": self.min_sample,
            "categories": stats,
            "how": ("counted from markets the venue reports as closed, where "
                    "YES means the first outcome settled at 1. A prior over "
                    "that venue's market population, not a forecast about one "
                    "question."),
        }
        usable = [c for c, v in stats.items() if v["usable"]]
        logger.info(
            f"Base-rate dataset built from {len(rows or [])} closed market(s): "
            f"{len(stats)} category/categories, {len(usable)} usable at "
            f"n>={self.min_sample}"
            + (" (" + ", ".join(
                   "{0} {1:.2f} n={2}".format(c, stats[c]["rate"], stats[c]["n"])
                   for c in usable[:5]) + ")"
               if usable else " - none has a large enough sample yet"))
        return self.dataset

    # -- use ---------------------------------------------------------------
    def prior_for(self, category: str) -> Optional[Dict[str, Any]]:
        """
        The prior for a category, or None when there is no usable sample.

        None is a real answer: it means the model must contribute no weight
        rather than a constant dressed up as a frequency.
        """
        if not self.dataset:
            return None
        entry = (self.dataset.get("categories") or {}).get(
            str(category or "default").lower())
        if not entry or not entry.get("usable") or not entry.get("n"):
            return None
        return {
            "category": str(category).lower(),
            "rate": float(entry["rate"]),
            "n": int(entry["n"]),
            "mean_last_price": entry.get("mean_last_price"),
            "source": self.dataset.get("source", ""),
            "built_at": self.dataset.get("built_at", ""),
        }

    def status(self) -> Dict[str, Any]:
        """What the console shows: is this real evidence yet, and from what."""
        if not self.dataset:
            return {
                "available": False,
                "reason": ("no base-rate dataset yet - the base-rate model "
                           "contributes no weight until real resolved markets "
                           "have been counted"),
                "min_sample": self.min_sample,
            }
        categories = self.dataset.get("categories") or {}
        return {
            "available": any(v.get("usable") for v in categories.values()),
            "built_at": self.dataset.get("built_at", ""),
            "source": self.dataset.get("source", ""),
            "markets_read": self.dataset.get("markets_read", 0),
            "markets_skipped": self.dataset.get("markets_skipped", 0),
            "min_sample": self.min_sample,
            "categories": categories,
            "how": self.dataset.get("how", ""),
        }


def prior_from_context(context: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
    """
    Pull the prior the cycle loaded into a context dict.

    Kept as a function so the model can be given a prior by any caller - a test,
    a backtest, or the live cycle - through one shape.
    """
    book = (context or {}).get("base_rate_prior")
    if isinstance(book, dict):
        if book.get("rate") is not None and int(book.get("n") or 0) > 0:
            return book
    return None
