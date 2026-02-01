"""
NeuroTrader Model Registry
==========================
Versioned model storage with metadata tracking.

Features:
- Automatic versioning (v001, v002, ...)
- Metadata tracking (config, metrics, timestamps)
- Best model promotion (symlink/copy)
- Model comparison utilities
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    version: int
    role: str
    symbol: str
    timeframe: str
    created_at: str
    training_steps: int
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    data_path: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ModelMetadata':
        return cls(**d)


class ModelRegistry:
    """
    Central registry for trained models with versioning.
    
    Directory Structure:
        models/
        ├── registry.json          # Master index
        ├── scalper/
        │   ├── v001/
        │   │   ├── model.zip
        │   │   ├── metadata.json
        │   │   └── backtest.csv (optional)
        │   ├── v002/
        │   └── best/              # Copy of best version
        ├── swing/
        └── trend/
    """
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.registry_file = self.base_dir / "registry.json"
        self._ensure_structure()
        self._load_registry()
        
    def _ensure_structure(self):
        """Create directory structure if not exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for role in ["scalper", "swing", "trend"]:
            (self.base_dir / role).mkdir(exist_ok=True)
            
    def _load_registry(self):
        """Load or create master registry index."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "created_at": datetime.now().isoformat(),
                "models": {},
                "best": {}
            }
            self._save_registry()
            
    def _save_registry(self):
        """Save registry index to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
            
    def _get_next_version(self, role: str) -> int:
        """Get next version number for a role."""
        role_dir = self.base_dir / role
        existing = [d.name for d in role_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
        if not existing:
            return 1
        versions = [int(v[1:]) for v in existing if v[1:].isdigit()]
        return max(versions) + 1 if versions else 1
    
    def register_model(
        self,
        model_path: str,
        role: str,
        symbol: str,
        timeframe: str,
        training_steps: int,
        training_config: Dict[str, Any],
        metrics: Dict[str, float],
        data_path: str,
        backtest_csv: Optional[str] = None
    ) -> ModelMetadata:
        """
        Register a new model version.
        
        Args:
            model_path: Path to the trained model file (.zip)
            role: Agent role (scalper/swing/trend)
            symbol: Trading symbol (e.g., XAUUSD)
            timeframe: Data timeframe (e.g., M5)
            training_steps: Total training steps
            training_config: Hyperparameters used
            metrics: Evaluation metrics (return, sharpe, etc.)
            data_path: Path to training data
            backtest_csv: Optional path to backtest results
            
        Returns:
            ModelMetadata for the registered model
        """
        version = self._get_next_version(role)
        version_str = f"v{version:03d}"
        version_dir = self.base_dir / role / version_str
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        model_dest = version_dir / "model.zip"
        shutil.copy2(model_path, model_dest)
        
        # Copy backtest if provided
        if backtest_csv and os.path.exists(backtest_csv):
            shutil.copy2(backtest_csv, version_dir / "backtest.csv")
        
        # Create metadata
        metadata = ModelMetadata(
            version=version,
            role=role,
            symbol=symbol,
            timeframe=timeframe,
            created_at=datetime.now().isoformat(),
            training_steps=training_steps,
            training_config=training_config,
            metrics=metrics,
            data_path=data_path
        )
        
        # Save metadata
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update registry index
        model_key = f"{role}/{version_str}"
        self.registry["models"][model_key] = {
            "path": str(version_dir),
            "metrics": metrics,
            "created_at": metadata.created_at
        }
        self._save_registry()
        
        print(f"✅ Registered: {model_key}")
        return metadata
    
    def promote_best(self, role: str, version: int, metric_name: str = None) -> bool:
        """
        Promote a version to 'best' for its role.
        
        Args:
            role: Agent role
            version: Version number to promote
            metric_name: Optional metric to log as reason
            
        Returns:
            True if successful
        """
        version_str = f"v{version:03d}"
        source_dir = self.base_dir / role / version_str
        best_dir = self.base_dir / role / "best"
        
        if not source_dir.exists():
            print(f"❌ Version {version_str} not found for {role}")
            return False
        
        # Remove existing best
        if best_dir.exists():
            shutil.rmtree(best_dir)
        
        # Copy to best
        shutil.copytree(source_dir, best_dir)
        
        # Update registry
        self.registry["best"][role] = {
            "version": version,
            "promoted_at": datetime.now().isoformat(),
            "metric": metric_name
        }
        self._save_registry()
        
        print(f"🏆 Promoted {role}/{version_str} to BEST")
        return True
    
    def get_best(self, role: str) -> Optional[Dict[str, Any]]:
        """Get info about the best model for a role."""
        best_dir = self.base_dir / role / "best"
        if not best_dir.exists():
            return None
        
        metadata_file = best_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_best_model_path(self, role: str) -> Optional[str]:
        """Get path to the best model.zip for a role."""
        best_model = self.base_dir / role / "best" / "model.zip"
        return str(best_model) if best_model.exists() else None
    
    def list_versions(self, role: str) -> List[Dict[str, Any]]:
        """List all versions for a role with summary info."""
        role_dir = self.base_dir / role
        versions = []
        
        for version_dir in sorted(role_dir.iterdir()):
            if not version_dir.is_dir() or not version_dir.name.startswith('v'):
                continue
            
            metadata_file = version_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                    versions.append({
                        "version": version_dir.name,
                        "created_at": meta.get("created_at"),
                        "metrics": meta.get("metrics", {})
                    })
        
        return versions
    
    def compare(self, role: str, versions: List[int] = None) -> str:
        """
        Compare metrics across versions.
        
        Args:
            role: Agent role
            versions: Specific versions to compare (None = all)
            
        Returns:
            Formatted comparison table
        """
        all_versions = self.list_versions(role)
        
        if versions:
            version_strs = {f"v{v:03d}" for v in versions}
            all_versions = [v for v in all_versions if v["version"] in version_strs]
        
        if not all_versions:
            return f"No versions found for {role}"
        
        # Build comparison table
        lines = [
            f"\n📊 Model Comparison: {role.upper()}",
            "=" * 60
        ]
        
        # Header
        header = f"{'Version':<10} {'Created':<20} {'Return':<12} {'Sharpe':<10}"
        lines.append(header)
        lines.append("-" * 60)
        
        for v in all_versions:
            metrics = v.get("metrics", {})
            created = v.get("created_at", "")[:16]  # Truncate timestamp
            ret = metrics.get("total_return", metrics.get("return_pct", 0))
            sharpe = metrics.get("sharpe_ratio", 0)
            
            line = f"{v['version']:<10} {created:<20} {ret:>10.2f}% {sharpe:>8.2f}"
            lines.append(line)
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def auto_promote_if_better(
        self,
        role: str,
        new_version: int,
        primary_metric: str,
        higher_is_better: bool = True
    ) -> bool:
        """
        Automatically promote new version if it beats current best.
        
        Args:
            role: Agent role
            new_version: Version to evaluate
            primary_metric: Metric name to compare
            higher_is_better: Whether higher values are better
            
        Returns:
            True if promoted
        """
        new_version_str = f"v{new_version:03d}"
        new_meta_file = self.base_dir / role / new_version_str / "metadata.json"
        
        if not new_meta_file.exists():
            return False
        
        with open(new_meta_file, 'r') as f:
            new_meta = json.load(f)
        new_value = new_meta.get("metrics", {}).get(primary_metric, 0)
        
        # Get current best
        current_best = self.get_best(role)
        if current_best is None:
            # No best exists, promote automatically
            self.promote_best(role, new_version, primary_metric)
            return True
        
        current_value = current_best.get("metrics", {}).get(primary_metric, 0)
        
        # Compare
        is_better = new_value > current_value if higher_is_better else new_value < current_value
        
        if is_better:
            print(f"📈 New best! {primary_metric}: {current_value:.2f} → {new_value:.2f}")
            self.promote_best(role, new_version, primary_metric)
            return True
        else:
            print(f"📉 Not better. Current best {primary_metric}: {current_value:.2f}, New: {new_value:.2f}")
            return False


# Quick test
if __name__ == "__main__":
    registry = ModelRegistry("models")
    
    # Example usage
    print(registry.compare("scalper"))
    print("\nBest Scalper:", registry.get_best("scalper"))
