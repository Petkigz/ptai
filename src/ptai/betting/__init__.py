"""
Betting / sports exchange subsystem.

Real odds maths, real feeds, exchange back/lay semantics, arbitrage with
account-risk management, and Poisson/Elo models blended against the market.

Design rule for the whole package: it never invents a price. If a feed is
unreachable the result is empty plus a diagnostic, because a fabricated
line is worse than no line - everything downstream treats it as real and
sizes money against it.
"""
from .odds_math import (
    ArbitrageLeg,
    ArbitrageOpportunity,
    BookMetrics,
    DutchResult,
    american_to_decimal,
    analyse_book,
    betfair_tick_size,
    closing_line_value,
    commission_adjusted_decimal,
    decimal_to_american,
    devig,
    dutch,
    expected_value,
    fair_decimal_odds,
    find_arbitrage,
    implied_probability,
    is_sharp_process,
    kelly_fraction,
    kelly_stake,
    lay_liability,
    parse_odds,
    snap_to_ladder,
)
from .sports_data import (
    BaseProvider,
    BookOdds,
    ConsensusOdds,
    EspnProvider,
    FootballDataProvider,
    LineMovement,
    SportsDataEngine,
    SportsEvent,
    TheOddsApiProvider,
    sample_book,
    sample_events,
)
from .models import (
    BASE_RATES,
    EloForecast,
    SoccerForecast,
    SportsForecast,
    SportsModelEngine,
    dixon_coles_tau,
    elo_expected,
    elo_forecast,
    elo_update,
    poisson_soccer,
    scoreline_matrix,
    shrink_to_base_rate,
)
from .exchange import (
    BetSide,
    ExchangeAccount,
    ExchangeBet,
    ExchangeBook,
    ExchangeMarketMaker,
    ExchangePosition,
    OrderStatus,
    PriceLevel,
    Quote,
    size_back,
    size_lay,
)
from .arb import (
    BookProfile,
    BookmakerRiskManager,
    ExchangeLayArb,
    book_vs_exchange_arb,
    dutch_outcome_group,
    hedge_to_green,
    match_events,
    multi_book_arb,
    normalise_team,
)
from .market_types import (
    MARKET_CATALOGUE,
    MARKET_STAT,
    MatchFacts,
    MarketSpec,
    Settlement,
    catalogue_report,
    get_spec,
    markets_for_sport,
    markets_needing_model,
    payout_multiplier,
    settle_market,
    validate_line,
)
from .derivative_markets import (
    DEFAULT_CARDS_TOTAL,
    DEFAULT_CORNERS_AWAY,
    DEFAULT_CORNERS_HOME,
    DEFAULT_REDS_PER_MATCH,
    AccumulatorLeg,
    AccumulatorQuote,
    EventMarketPrices,
    GoalMarketPrices,
    HalfTimePrices,
    MatchCardPrices,
    price_accumulator,
    price_anytime_scorer,
    price_booking_points,
    price_count_market,
    price_first_scorer,
    price_half_time,
    price_match_card,
    price_player_count,
    price_red_card,
)
from .high_scoring import (
    LEAGUE_TO_SPORT,
    SPORT_PARAMS,
    resolve_sport,
    HighScoringCard,
    SportParams,
    TotalPrices,
    SpreadPrices,
    is_high_scoring,
    normal_cdf,
    params_for,
    price_high_scoring_card,
    price_margin_band,
    price_margin_bands,
    price_moneyline_normal,
    price_period_spread,
    price_period_total,
    price_race_to_points,
    price_spread_normal,
    price_team_total_normal,
    price_total_normal,
)
from .engine import BetOpportunity, BettingEngine

__all__ = [
    # odds math
    "ArbitrageLeg", "ArbitrageOpportunity", "BookMetrics", "DutchResult",
    "american_to_decimal", "analyse_book", "betfair_tick_size", "closing_line_value",
    "commission_adjusted_decimal", "decimal_to_american", "devig", "dutch",
    "expected_value", "fair_decimal_odds", "find_arbitrage", "implied_probability",
    "is_sharp_process", "kelly_fraction", "kelly_stake", "lay_liability",
    "parse_odds", "snap_to_ladder",
    # data
    "BaseProvider", "BookOdds", "ConsensusOdds", "EspnProvider", "FootballDataProvider",
    "LineMovement", "SportsDataEngine", "SportsEvent", "TheOddsApiProvider",
    "sample_book", "sample_events",
    # models
    "BASE_RATES", "EloForecast", "SoccerForecast", "SportsForecast", "SportsModelEngine",
    "dixon_coles_tau", "elo_expected", "elo_forecast", "elo_update",
    "poisson_soccer", "scoreline_matrix", "shrink_to_base_rate",
    # exchange
    "BetSide", "ExchangeAccount", "ExchangeBet", "ExchangeBook", "ExchangeMarketMaker",
    "ExchangePosition", "OrderStatus", "PriceLevel", "Quote", "size_back", "size_lay",
    # arb
    "BookProfile", "BookmakerRiskManager", "ExchangeLayArb", "book_vs_exchange_arb",
    "dutch_outcome_group", "hedge_to_green", "match_events", "multi_book_arb",
    "normalise_team",
    # market catalogue & settlement
    "MARKET_CATALOGUE", "MARKET_STAT", "MatchFacts", "MarketSpec", "Settlement",
    "catalogue_report", "get_spec", "markets_for_sport", "markets_needing_model",
    "payout_multiplier", "settle_market", "validate_line",
    # derivative pricing
    "DEFAULT_CARDS_TOTAL", "DEFAULT_CORNERS_AWAY", "DEFAULT_CORNERS_HOME",
    "DEFAULT_REDS_PER_MATCH", "AccumulatorLeg", "AccumulatorQuote",
    "EventMarketPrices", "GoalMarketPrices", "HalfTimePrices", "MatchCardPrices",
    "price_accumulator", "price_anytime_scorer", "price_booking_points",
    "price_count_market", "price_first_scorer", "price_half_time",
    "price_match_card", "price_player_count", "price_red_card",
    # high-scoring sports (normal margin model, not Poisson)
    "LEAGUE_TO_SPORT", "resolve_sport",
    "SPORT_PARAMS", "HighScoringCard", "SportParams", "TotalPrices", "SpreadPrices",
    "is_high_scoring", "normal_cdf", "params_for", "price_high_scoring_card",
    "price_margin_band", "price_margin_bands", "price_moneyline_normal",
    "price_period_spread", "price_period_total", "price_race_to_points",
    "price_spread_normal", "price_team_total_normal", "price_total_normal",
    # engine
    "BetOpportunity", "BettingEngine",
]
