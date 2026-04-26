import sys
from src.core.pipeline import QuantPipeline

def main():
    """
    @brief Tool entry point.
    """
    # Create the pipeline instance
    pipeline = QuantPipeline(config_path="input/configuration.yaml")
    
    # Run the default stock analysis mode
    success = pipeline.run(mode="stock_analysis")
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
