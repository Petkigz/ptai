"""Market-agnostic venue adapters - expanded beyond Polymarket"""
from .adapter import MarketAdapter, AdapterCapability, VenueType, EligibilityStatus, VenueOpportunity
from .registry import VenueRegistry
from .polymarket_adapter import PolymarketAdapter
from .kalshi_adapter import KalshiAdapter
from .manifold_adapter import ManifoldAdapter
from .crypto_adapter import CryptoAdapter
from .stock_adapter import StockAdapter
from .predictit_adapter import PredictItAdapter
from .simmer_adapter import SimmerAdapter
from .cymetica_adapter import CymeticaAdapter
from .whitebit_adapter import WhiteBITAdapter
from .afx_adapter import AFXAdapter
from .grvt_adapter import GRVTAdapter
from .pionex_adapter import PionexAdapter
from .betfair_adapter import BetfairAdapter, BetdaqAdapter, BetConnectAdapter
from .ccxt_adapter import CCXTUnifiedAdapter
from .veynor_adapter import VeynorAdapter
from .openpx_adapter import OpenPXAdapter
from .apify_adapter import ApifyAdapter

__all__ = [
    "MarketAdapter", "AdapterCapability", "VenueType", "EligibilityStatus", "VenueOpportunity",
    "VenueRegistry",
    "PolymarketAdapter", "KalshiAdapter", "ManifoldAdapter", "CryptoAdapter", "StockAdapter",
    "PredictItAdapter", "SimmerAdapter", "CymeticaAdapter",
    "WhiteBITAdapter", "AFXAdapter", "GRVTAdapter", "PionexAdapter",
    "BetfairAdapter", "BetdaqAdapter", "BetConnectAdapter",
    "CCXTUnifiedAdapter", "VeynorAdapter", "OpenPXAdapter", "ApifyAdapter"
]
