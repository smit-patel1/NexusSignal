#!/usr/bin/env python
"""
NexusSignal 2.0 Training Script

Usage:
    python -m nexus2.run_training                    # Train all tickers with default config
    python -m nexus2.run_training --ticker AAPL     # Train single ticker
    python -m nexus2.run_training --config my.yaml  # Use custom config

This script:
1. Loads configuration
2. Builds features with fractional differencing
3. Generates Triple Barrier labels
4. Trains barrier classifier with CPCV
5. Applies meta-labeling
6. Generates trading signals
7. Saves model artifacts
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus2.pipeline.trainer import NexusTrainer, main

if __name__ == "__main__":
    main()

