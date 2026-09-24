"""
Capital - where the money is, and which money the agent is allowed to risk.

THE ANSWER TO "HOW DO I GET CAPITAL INTO THIS SYSTEM" IS: YOU DO NOT SEND THE
SYSTEM ANYTHING. There is no account here for it to hold, no deposit address it
owns, and no key it needs for your money to exist. The money stays at the venue,
in YOUR account. What this module holds is a RECORD of how much of it the agent
may use.

Why it has to be that way, because it decides the whole design:

  * A venue pays a winning position into the wallet that holds it. If the agent
    traded through an account it does not control, it could not redeem, could
    not see the balance, and could not settle. Concretely, Polymarket deposits
    are USDC on Polygon converted to pUSD, and redemption returns collateral to
    the wallet that owns the shares.
  * So the agent's capital IS the venue balance. One login, one pool of money,
    and the agent's job is to not spend it twice.

That last part is the reason this module exists. Left alone, the agent will count
the same $50 three times:

    1. it reads a venue balance of $50 and sizes 6% of it;
    2. it opens four positions with that $50 and sizes 6% of $50 again;
    3. it places a resting order, which reserves cash at the venue but shows as
       no position at all.

Step 3 is the one nothing else covers. An unfilled order is invisible to a
position ledger, so without a reservation on top, the agent sizes a fifth trade
against money the venue has already locked behind the fourth order.

So every account carries four numbers, and they must reconcile:

    deposited  = realised into the account
    in_positions   = cost of open positions
    reserved       = cash locked behind working orders
    available      = deposited - in_positions - reserved    (never negative)

`available` is what sizing reads. Not the balance field, not equity.

Two more decisions worth stating plainly:

  * The agent never holds a key to your whole wallet. Give it a DEDICATED wallet,
    funded with the amount you are willing to lose, and nothing else. That is the
    real budget limit - it is enforced by the chain, not by a config value the
    agent could get wrong.
  * A $50 budget cannot be spread across many venues. Most venues have a minimum
    order size, and several have a minimum DEPOSIT, so the honest plan is one
    venue live and the rest in paper mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# Order in which venues are worth funding with a small budget. The reason is not
# quality of opportunity, it is the cost of getting money in and back out:
# settling a prediction market returns USDC to the same wallet in minutes, and
# the deposit path is a USDC transfer rather than a bank wire.
FUNDING_ROUTES = {
    "polymarket": {
        "label": "Polymarket",
        "currency": "USDC on Polygon (converted to pUSD on deposit)",
        # Who can actually open this account. Recorded so venue selection can
        # rank on it rather than picking alphabetically.
        "available_from": "global, excluding restricted jurisdictions",
        "residency_required": None,
        "minimum_deposit_usd": 1.0,
        "recommended_deposit_usd": 20.0,
        "smallest_practical_usd": 10.0,
        "deposit_steps": [
            "Create and verify a Polymarket account (KYC/ID where required).",
            "In Polymarket: balance -> Deposit -> Deposit Crypto. Copy the "
            "Polygon USDC address it shows. That address is YOURS; the agent "
            "never sees it and never needs it.",
            "Send USDC on the Polygon network only. Wrong network means lost "
            "funds - this is the single most common way people lose money here.",
            "From an exchange (Coinbase, Kraken, Binance): withdraw USDC, "
            "network = Polygon, address = the one Polymarket showed you. "
            "Roughly free and settles in minutes.",
            "From a card: Polymarket's built-in on-ramp (MoonPay/Transak). "
            "Fastest and simplest, but costs about 2-5% and usually has a "
            "$30-50 minimum, which matters on a $50 budget.",
            "Wait for the balance to appear. Deposits convert to pUSD "
            "automatically.",
        ],
        "withdraw_note": (
            "Withdrawals go back to a Polygon address you control. Positions "
            "must be redeemed before the money is available to withdraw."
        ),
        "fees_note": (
            "Deposits and withdrawals cost Polygon gas (cents). Trading is "
            "gasless: orders are signed off-chain, and settlement/redemption "
            "rides the gasless relayer."
        ),
        "agent_key_explanation": (
            "The agent trades through the account. It needs a signer key for "
            "that account - a dedicated wallet, funded with only the budget you "
            "are willing to lose. It never needs your seed phrase."
        ),
    },
    "kalshi": {
        "label": "Kalshi",
        "currency": "USD (regulated US exchange)",
        "available_from": "United States only",
        "residency_required": "US bank account and US identity verification",
        "minimum_deposit_usd": 10.0,
        "recommended_deposit_usd": 50.0,
        "smallest_practical_usd": 25.0,
        "deposit_steps": [
            "Kalshi is USD-denominated and regulated in the US, with its own "
            "signup and KYC.",
            "Fund by ACH or debit card from inside Kalshi. ACH is free and "
            "takes 1-3 business days; debit is instant for a small fee.",
            "There is no crypto step, which makes it the simpler option for a "
            "first deposit - and the wrong option if you are outside the US.",
        ],
        "withdraw_note": "Withdraw back to the same bank account, typically 1-3 business days.",
        "fees_note": "Built-in trading fees are charged per contract, not per deposit.",
        "agent_key_explanation": (
            "Kalshi uses an API key plus a downloaded private key rather than a "
            "wallet seed. Generate both in the Kalshi account settings."
        ),
    },
}

# Venues that are scanned for opportunity but cannot be funded by a small
# account, with the reason. Recorded so the UI can say WHY rather than showing an
# empty row.
UNFUNDABLE_SMALL = {
    "kalshi": "US-regulated: needs KYC and a US bank account",
    "manifold": "play-money only: no real capital can be deployed",
    "crypto_binance": "needs an exchange account, KYC, and API keys",
    "stock_mock": "simulated data only",
    "afx_dex": "on-chain DEX: needs MATIC for gas and a funded wallet",
}


@dataclass
class VenueAccount:
    """One venue account, and the part of its money the agent may use."""

    venue_id: str
    venue_label: str = ""
    funded: bool = False
    # What the venue says the balance is.
    reported_balance_usd: float = 0.0
    balance_is_real: bool = False
    # How much of it the operator has authorised the agent to use. Zero means
    # the agent may trade here in paper mode only.
    budget_usd: float = 0.0
    # Committed money.
    in_positions_usd: float = 0.0
    reserved_usd: float = 0.0
    # Bookkeeping.
    deposited_usd: float = 0.0
    realised_pnl_usd: float = 0.0
    currency: str = ""
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def available_usd(self) -> float:
        """
        What sizing may use. Never negative.

        A negative figure would mean the agent has committed more than it has,
        which is a state it must not be able to enter and must be told about
        loudly if it does.
        """
        return max(0.0, self.deposited_usd - self.in_positions_usd - self.reserved_usd)

    @property
    def committed_usd(self) -> float:
        return self.in_positions_usd + self.reserved_usd

    @property
    def deployment_pct(self) -> float:
        if self.deposited_usd <= 0:
            return 0.0
        return self.committed_usd / self.deposited_usd * 100.0

    @property
    def over_committed(self) -> bool:
        return self.committed_usd > self.deposited_usd + 1e-9

    @property
    def can_deploy_live(self) -> bool:
        """Live capital here needs an authorised budget AND free cash."""
        return self.funded and self.budget_usd > 0 and self.available_usd > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "venue_label": self.venue_label or self.venue_id,
            "funded": self.funded,
            "reported_balance_usd": round(self.reported_balance_usd, 2),
            "balance_is_real": self.balance_is_real,
            "budget_usd": round(self.budget_usd, 2),
            "deposited_usd": round(self.deposited_usd, 2),
            "in_positions_usd": round(self.in_positions_usd, 2),
            "reserved_usd": round(self.reserved_usd, 2),
            "committed_usd": round(self.committed_usd, 2),
            "available_usd": round(self.available_usd, 2),
            "realised_pnl_usd": round(self.realised_pnl_usd, 2),
            "deployment_pct": round(self.deployment_pct, 1),
            "can_deploy_live": self.can_deploy_live,
            "over_committed": self.over_committed,
            "currency": self.currency,
            "notes": self.notes,
            "warnings": self.warnings,
        }


@dataclass
class CapitalPlan:
    """
    The operator's answer to "how much, and where".

    There is deliberately no "transfer money to the agent" step, because there is
    nowhere for the money to go. The operator deposits at the venue and records
    the budget here.
    """

    total_budget_usd: float = 0.0
    allocations: Dict[str, float] = field(default_factory=dict)
    mode: str = "paper"
    accounts: List[VenueAccount] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def allocated_usd(self) -> float:
        return sum(self.allocations.values())

    @property
    def unallocated_usd(self) -> float:
        return round(self.total_budget_usd - self.allocated_usd, 6)

    @property
    def live_venues(self) -> List[str]:
        return [a.venue_id for a in self.accounts if a.can_deploy_live]

    @property
    def total_available_usd(self) -> float:
        return round(sum(a.available_usd for a in self.accounts), 6)

    @property
    def total_reserved_usd(self) -> float:
        return round(sum(a.reserved_usd for a in self.accounts), 6)

    @property
    def total_in_positions_usd(self) -> float:
        return round(sum(a.in_positions_usd for a in self.accounts), 6)

    @property
    def is_deployable(self) -> bool:
        """
        Fail closed. No funded, budgeted venue with free cash means paper only -
        the same rule the qualification gate enforces, applied to money.
        """
        return self.mode == "live" and bool(self.live_venues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "total_budget_usd": round(self.total_budget_usd, 2),
            "allocated_usd": round(self.allocated_usd, 2),
            "unallocated_usd": round(self.unallocated_usd, 2),
            "allocations": {k: round(v, 2) for k, v in self.allocations.items()},
            "accounts": [a.to_dict() for a in self.accounts],
            "live_venues": self.live_venues,
            "total_available_usd": self.total_available_usd,
            "total_reserved_usd": self.total_reserved_usd,
            "total_in_positions_usd": self.total_in_positions_usd,
            "is_deployable": self.is_deployable,
            "warnings": self.warnings,
        }


class CapitalLedger:
    """
    Assembles the plan from what is actually recorded, and refuses to invent.

    Every figure here traces to a row: a position cost from the trades table, a
    reservation from the orders table, a budget from the operator's own setting.
    If the venue's balance cannot be read, `balance_is_real` is False and the
    account is not treated as funded - a guessed balance is how an agent decides
    it has money it does not have.
    """

    def __init__(self, storage=None, execution=None, order_manager=None,
                 settings=None):
        self.storage = storage
        self.execution = execution
        self.order_manager = order_manager
        self.settings = settings

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------

    def build(self, mode: str = "paper",
              budgets: Optional[Dict[str, float]] = None,
              venue_labels: Optional[Dict[str, str]] = None,
              balances: Optional[Dict[str, Dict[str, Any]]] = None) -> CapitalPlan:
        """
        The current capital position, per venue.

        `balances` is what the venue reported, keyed by venue id, in the shape
        {"venue": {"available": bool, "balance": float, "source": str}}. Absent
        or unavailable means the account is NOT funded as far as this ledger is
        concerned, whatever the operator typed into a budget field.
        """
        budgets = budgets or {}
        venue_labels = venue_labels or {}
        balances = balances or {}
        plan = CapitalPlan(mode=mode,
                           total_budget_usd=round(sum(budgets.values()), 6),
                           allocations=dict(budgets))

        for venue_id, budget in sorted(budgets.items()):
            account = self._account(venue_id, budget, venue_labels.get(venue_id, venue_id),
                                    balances.get(venue_id))
            plan.accounts.append(account)

        # A venue with money in it but no recorded budget still has to appear, or
        # its reservations are invisible.
        for venue_id, balance in balances.items():
            if venue_id in budgets:
                continue
            if not (balance or {}).get("available"):
                continue
            real = float((balance or {}).get("balance") or 0.0)
            if real <= 0:
                continue
            account = self._account(venue_id, 0.0,
                                    venue_labels.get(venue_id, venue_id), balance)
            account.warnings.append(
                f"the venue reports ${real:.2f}; no budget has been authorised "
                f"for it, so the agent will not deploy capital here")
            plan.accounts.append(account)

        if mode == "live" and not plan.live_venues:
            plan.warnings.append(
                "live mode requested but no venue is both funded and authorised: "
                "every venue runs in paper mode until a budget is set")
        for account in plan.accounts:
            if account.over_committed:
                plan.warnings.append(
                    f"{account.venue_id} has ${account.committed_usd:.2f} committed "
                    f"against ${account.deposited_usd:.2f} deposited - more capital "
                    f"is committed than exists")
        return plan

    def _account(self, venue_id: str, budget: float, label: str,
                 balance: Optional[Dict[str, Any]]) -> VenueAccount:
        account = VenueAccount(venue_id=venue_id, venue_label=label,
                               budget_usd=float(budget or 0.0))
        route = FUNDING_ROUTES.get(venue_id, {})
        account.currency = route.get("currency", "")

        balance = balance or {}
        account.balance_is_real = bool(balance.get("available"))
        account.reported_balance_usd = float(balance.get("balance") or 0.0) \
            if account.balance_is_real else 0.0

        # Deposited capital is the authorised budget, not the venue's balance.
        # The balance may hold money from other activity, or money the operator
        # does not want this agent touching.
        account.deposited_usd = account.budget_usd

        positions = self._position_cost(venue_id)
        reservations = self._reserved(venue_id)
        account.in_positions_usd = positions
        account.reserved_usd = reservations
        account.realised_pnl_usd = self._realised(venue_id)

        # Funded means "the venue answered AND we are authorised to use it".
        # A budget without a venue balance is a plan, not money; a balance
        # without a budget is money the agent has not been given permission to
        # touch. Neither alone can deploy live capital.
        account.funded = account.balance_is_real and account.budget_usd > 0

        if not account.balance_is_real:
            account.warnings.append(
                "the venue's balance could not be read, so this account is not "
                "treated as funded. A guessed balance is how an agent decides "
                "it has money it does not have.")
        elif account.budget_usd <= 0:
            account.notes.append(
                "no budget authorised for this venue: paper trading only here")
        elif account.budget_usd > account.reported_balance_usd + 1e-9:
            account.warnings.append(
                f"authorised budget ${account.budget_usd:.2f} exceeds the "
                f"venue's reported balance ${account.reported_balance_usd:.2f}; "
                f"sizing uses the smaller figure")
            account.deposited_usd = min(account.budget_usd,
                                        account.reported_balance_usd)
        else:
            account.notes.append(
                f"authorised ${account.budget_usd:.2f} of a reported "
                f"${account.reported_balance_usd:.2f} balance")

        if account.reserved_usd > 0:
            account.notes.append(
                f"${account.reserved_usd:.2f} is locked behind working orders "
                f"and is not available to size against")
        return account

    # ------------------------------------------------------------------
    # what is actually committed, by venue
    # ------------------------------------------------------------------

    def _position_cost(self, venue_id: str) -> float:
        """
        Cost of positions still open at this venue.

        Live and paper positions are separate pools: a paper position must not
        consume live capital, and a live position must not be hidden by paper
        ones. They are summed separately and only the live figure is charged to
        the live budget.
        """
        if self.storage is None:
            return 0.0
        total = 0.0
        try:
            for position in self.storage.get_open_positions() or []:
                if str(position.get("venue_id") or "") != venue_id:
                    continue
                if str(position.get("status") or "").lower() == "paper":
                    continue
                total += float(position.get("position_size_usd") or 0.0)
        except Exception as e:
            logger.error(f"Could not total positions for {venue_id}: "
                         f"{type(e).__name__}: {e}")
            return 0.0
        return round(total, 6)

    def _reserved(self, venue_id: str) -> float:
        """
        Cash locked behind orders that have not filled.

        Its own figure, never merged into positions: a resting order has bought
        nothing, so it is not a position - but it has committed the cash, so it
        is not available either. Adding it to both, or to neither, is how the
        same dollars get spent twice.
        """
        if self.storage is None:
            return 0.0
        try:
            return round(float(self.storage.resting_capital_usd(venue_id=venue_id)), 6)
        except Exception as e:
            logger.error(f"Could not total reserved capital for {venue_id}: "
                         f"{type(e).__name__}: {e}")
            return 0.0

    def _realised(self, venue_id: str) -> float:
        if self.storage is None:
            return 0.0
        try:
            row = self.storage.conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS total FROM trades "
                "WHERE resolved = 1 AND venue_id = ?", (venue_id,)).fetchone()
            return round(float(row["total"] or 0.0), 6)
        except Exception as e:
            logger.warning(f"Could not total realised P&L for {venue_id}: {e}")
            return 0.0


# ----------------------------------------------------------------------
# the operator's plan, sized to the budget
# ----------------------------------------------------------------------

def plan_for_budget(total_usd: float, mode: str = "paper",
                    preferred: Optional[str] = None) -> Dict[str, Any]:
    """
    Split a budget across venues, and say what it will not buy.

    One venue gets the money. On a small budget that is not a compromise, it is
    the correct answer: minimum order sizes, minimum deposits and per-venue
    withdrawal friction mean $50 spread over three accounts is three accounts
    that each cannot trade. `minimum_viable_usd` is the smallest amount worth
    depositing at all, and anything below it should stay in paper mode.

    Returns a plan, plus the reasons it is only a partial plan. It does not
    recommend depositing money - that is the operator's decision, and it depends
    on how much they are willing to lose.
    """
    total = float(total_usd or 0.0)
    primary = preferred or "polymarket"
    route = FUNDING_ROUTES.get(primary)
    result: Dict[str, Any] = {
        "mode": mode,
        "total_budget_usd": round(total, 2),
        "primary_venue": primary,
        "allocations": {},
        "notes": [],
        "warnings": [],
        "unfundable": dict(UNFUNDABLE_SMALL),
    }

    if route is None:
        result["warnings"].append(
            f"no funding route is recorded for {primary}; nothing can be "
            f"deposited there by this plan")
        return result

    smallest = float(route["smallest_practical_usd"])
    minimum = float(route["minimum_deposit_usd"])
    recommended = float(route["recommended_deposit_usd"])

    if total <= 0:
        result["warnings"].append(
            "budget is zero: run in paper mode. Paper mode needs no money at "
            "all and is where the strategy has to prove itself first.")
        return result

    if total < minimum:
        result["warnings"].append(
            f"${total:.2f} is below {route['label']}'s practical minimum "
            f"(${minimum:.2f}). Depositing less than that leaves an account that "
            f"cannot place a trade; keep it in paper mode instead.")
        return result

    if total < smallest:
        result["warnings"].append(
            f"${total:.2f} is above the deposit minimum but below the smallest "
            f"amount that can hold a position at the venue's minimum order size "
            f"(${smallest:.2f}). It will trade, but one position at a time.")

    result["allocations"][primary] = round(total, 2)
    result["notes"].append(
        f"All ${total:.2f} to {route['label']}. One venue, fully funded, beats "
        f"three venues that each cannot meet a minimum order.")
    if total >= recommended:
        result["notes"].append(
            f"${recommended:.2f} is the amount at which the venue's typical "
            f"minimum order size and the 6% position cap stop fighting each "
            f"other.")
    result["notes"].append(
        "The agent trades the account you fund. Deposit at the venue, not to "
        "the agent: it has no account of its own and never needs your seed phrase.")
    return result


# ----------------------------------------------------------------------
# the operator's authorisation, in one place
# ----------------------------------------------------------------------
#
# Mode and budget are the operator's permission to spend. They are read by the
# console (to show them), by the cycle (to size against them) and by the venue
# selection (to know which account is authorised). Three readers, so one
# definition: a second copy of these key names is how the screen and the trading
# loop end up disagreeing about how much money is authorised.

BUDGET_KEY_PREFIX = "console.budget."
MODE_KEY = "console.mode"


def authorised_budgets(storage) -> Dict[str, float]:
    """
    The budget the operator authorised, per venue.

    Only positive amounts are returned: a venue that is authorised $0 is not
    authorised, and returning it would make an unfunded venue look configured.
    """
    out: Dict[str, float] = {}
    if storage is None:
        return out
    for venue_id in FUNDING_ROUTES:
        raw = storage.get_state(f"{BUDGET_KEY_PREFIX}{venue_id}")
        try:
            value = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            # A budget that cannot be read is not a budget. Never guess.
            continue
        if value > 0:
            out[venue_id] = value
    return out


def authorised_budget(storage, venue_id: str) -> float:
    return float(authorised_budgets(storage).get(venue_id, 0.0))


def set_authorised_budget(storage, venue_id: str, amount_usd: float) -> None:
    storage.set_state(f"{BUDGET_KEY_PREFIX}{venue_id}", f"{float(amount_usd):.2f}")


def operator_mode(storage, default: str = "paper") -> str:
    """The mode the operator set. Paper unless they have said otherwise."""
    if storage is None:
        return default
    mode = (storage.get_state(MODE_KEY) or default).lower()
    return mode if mode in ("paper", "live") else default


def set_operator_mode(storage, mode: str) -> None:
    mode = str(mode).lower()
    if mode not in ("paper", "live"):
        raise ValueError(f"mode must be paper or live, got {mode!r}")
    storage.set_state(MODE_KEY, mode)
