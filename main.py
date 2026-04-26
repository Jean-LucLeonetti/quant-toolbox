import sys
from src.core.pipeline import QuantPipeline

def main():
    """
    @brief Tool entry point.
    """
    # Allow mode to be passed as a command line argument
    mode = "stock_analysis"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    # Create the pipeline instance
    pipeline = QuantPipeline(config_path="input/configuration.yaml")
    
    # Run the specified mode
    success = pipeline.run(mode=mode)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
