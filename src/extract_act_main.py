"""Main entry point for activation extraction."""
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from activation_processor import ActivationProcessor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    Main function for activation extraction.
    
    Args:
        config: Hydra configuration
    """
    print("Configuration:")
    print("\n" + OmegaConf.to_yaml(config))
    
    # Create processor and run
    processor = ActivationProcessor(config)
    processor.setup()
    processor.process()
    
    print("All done!")


if __name__ == "__main__":
    main()
