"""
Model Loader Utilities
=======================

Utilities for loading saved ML and RL models.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple
from stable_baselines3 import DQN, PPO, A2C

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading saved models"""
    
    def __init__(self, artifacts_root: Path):
        """
        Initialize model loader
        
        Args:
            artifacts_root: Root directory for artifacts
        """
        self.artifacts_root = Path(artifacts_root)
    
    def find_latest_ml_model(self) -> Optional[Path]:
        """Find the latest ML model file"""
        pattern = "**/ml_model*.pkl"
        matches = list(self.artifacts_root.glob(pattern))
        
        if not matches:
            # Also check reports directory (legacy location)
            reports_dir = self.artifacts_root.parent / "reports"
            if reports_dir.exists():
                matches = list(reports_dir.glob("ml_model*.pkl"))
        
        if not matches:
            return None
        
        # Return most recently modified
        return max(matches, key=lambda p: p.stat().st_mtime)
    
    def find_latest_rl_model(self) -> Optional[Path]:
        """Find the latest RL model file"""
        pattern = "**/rl_model*.zip"
        matches = list(self.artifacts_root.glob(pattern))
        
        if not matches:
            # Also check reports directory (legacy location)
            reports_dir = self.artifacts_root.parent / "reports"
            if reports_dir.exists():
                matches = list(reports_dir.glob("*_model*.zip"))
        
        if not matches:
            return None
        
        # Return most recently modified
        return max(matches, key=lambda p: p.stat().st_mtime)
    
    def load_ml_model(self, model_path: Optional[Path] = None) -> Optional[object]:
        """
        Load ML model (XGBoost classifier)
        
        Args:
            model_path: Path to model file (if None, finds latest)
        
        Returns:
            Loaded model object or None if not found
        """
        if model_path is None:
            model_path = self.find_latest_ml_model()
        
        if model_path is None:
            logger.warning("No ML model found")
            return None
        
        try:
            logger.info(f"Loading ML model from: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return None
    
    def load_rl_model(self, model_path: Optional[Path] = None, algorithm: str = 'DQN'):
        """
        Load RL model
        
        Args:
            model_path: Path to model file (if None, finds latest)
            algorithm: Algorithm type ('DQN', 'PPO', 'A2C')
        
        Returns:
            Loaded RL model or None if not found
        """
        if model_path is None:
            model_path = self.find_latest_rl_model()
        
        if model_path is None:
            logger.warning("No RL model found")
            return None
        
        try:
            logger.info(f"Loading RL model from: {model_path}")
            
            # Map algorithm string to class
            algorithm_map = {
                'DQN': DQN,
                'PPO': PPO,
                'A2C': A2C
            }
            
            algo_class = algorithm_map.get(algorithm.upper(), DQN)
            model = algo_class.load(str(model_path))
            
            logger.info(f"Successfully loaded {algorithm} model")
            return model
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """Get information about available models"""
        ml_path = self.find_latest_ml_model()
        rl_path = self.find_latest_rl_model()
        
        info = {
            'ml_model': {
                'found': ml_path is not None,
                'path': str(ml_path) if ml_path else None,
                'modified': ml_path.stat().st_mtime if ml_path else None
            },
            'rl_model': {
                'found': rl_path is not None,
                'path': str(rl_path) if rl_path else None,
                'modified': rl_path.stat().st_mtime if rl_path else None
            }
        }
        
        return info


def load_ml_scorer_components(model_data: dict):
    """
    Extract scorer components from saved model data
    
    Args:
        model_data: Dictionary containing model, scaler, feature_columns, etc.
    
    Returns:
        Tuple of (model, scaler, feature_columns, use_ml, baseline_scores)
    """
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    feature_columns = model_data.get('feature_columns', {})
    use_ml = model_data.get('use_ml', {})
    baseline_scores = model_data.get('baseline_scores', {})
    
    return model, scaler, feature_columns, use_ml, baseline_scores

