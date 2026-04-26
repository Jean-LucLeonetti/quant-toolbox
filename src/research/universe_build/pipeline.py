from src.data.universe import get_sp500_constituents, filter_universe_by_sector, fetch_ticker_metadata
from src.data.store import DataStore
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class UniversePipeline:
    """
    @brief Pipeline to build and refresh research universes.
    """
    def __init__(self):
        self.store = DataStore()

    def refresh(self, universes: list[str]):
        """
        @brief Refreshes the metadata for the specified universes.
        """
        logger.info(f"Starting universe refresh for: {universes}")
        
        # Always start by getting the latest S&P 500 list
        sp500_all = get_sp500_constituents()
        if sp500_all.empty:
            logger.error("Failed to fetch S&P 500 constituents. Aborting.")
            return False
            
        for uni in universes:
            logger.info(f"Processing universe: {uni}")
            
            if uni == "sp500_utilities_staples":
                # Universe 1: Utilities and Consumer Staples
                tickers = filter_universe_by_sector(sp500_all, ['Utilities', 'Consumer Staples'])
            elif uni == "sp500_utilities":
                # Sector focus: Utilities only
                tickers = filter_universe_by_sector(sp500_all, ['Utilities'])
            elif uni == "sp500":
                # Full S&P 500
                tickers = sp500_all['Symbol'].tolist()
            elif uni == "etf_pairs":
                # Structural ETF Pairs (legacy)
                tickers = ["GDX", "GDXJ", "USO", "UCO", "EWA", "EWC"]
            elif uni == "etf_sector":
                # Universe A: Sector ETFs
                tickers = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC", 
                          "VFH", "VHT", "VPU", "VDE"]
            elif uni == "etf_country":
                # Universe B: Country ETFs
                tickers = ["EWA", "EWC", "EWG", "EWH", "EWI", "EWJ", "EWL", "EWP", "EWQ", "EWS", 
                          "EWU", "EWW", "EWY", "EWZ", "EZA", "EIDO", "THD", "TUR"]
            elif uni == "etf_commodity":
                # Universe C: Commodity ETFs
                tickers = ["GLD", "SLV", "GDX", "GDXJ", "USO", "UCO", "UNG", "BOIL", 
                          "DBA", "CORN", "WEAT", "SOYB", "PPLT", "PALL", "URNM", "URA"]
            else:
                logger.warning(f"Unknown universe type: {uni}")
                continue
                
            # Fetch and store metadata
            metadata_df = fetch_ticker_metadata(tickers)
            self.store.upsert_tickers(metadata_df)
            
            # Update membership
            self.store.update_universe_membership(tickers, uni)
            
        logger.info("Universe refresh complete.")
        return True
