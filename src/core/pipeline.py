from src.research.stock_analysis.pipeline import run_stock_analysis
from src.research.universe_build.pipeline import UniversePipeline
from src.research.data_build.pipeline import DataBuildPipeline
from src.research.pairs.pipeline import PairsExplorationPipeline
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class QuantPipeline:
    """
    @brief Main orchestration class for the Quant toolbox.
    """
    def __init__(self, config_path: str = "input/configuration.yaml"):
        self.config_path = config_path

    def run(self, mode: str = "stock_analysis"):
        """
        @brief Runs the specified analysis mode.
        
        @param mode The analysis mode to run (e.g., 'stock_analysis', 'pairs').
        """
        logger.info(f"Starting Quant Pipeline in '{mode}' mode...")
        
        if mode == "stock_analysis":
            success = run_stock_analysis(self.config_path)
        elif mode == "universe_build":
            pipeline = UniversePipeline()
            # Refresh our core universes
            success = pipeline.refresh(["sp500", "sp500_utilities", "sp500_utilities_staples"])
        elif mode == "data_build":
            pipeline = DataBuildPipeline()
            # Default to sp500_utilities if no config provided
            success = pipeline.build("sp500_utilities", "2015-01-01", "2025-12-31")
        elif mode == "pairs":
            pipeline = PairsExplorationPipeline(universe_name="sp500_utilities")
            success = pipeline.run(start="2015-01-01", end="2025-12-31")
        else:
            logger.error(f"Unknown analysis mode: {mode}")
            success = False
            
        if success:
            logger.info("Pipeline execution finished successfully.")
        else:
            logger.error("Pipeline execution failed.")
            
        return success
