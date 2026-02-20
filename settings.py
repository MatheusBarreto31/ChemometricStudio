"""Settings manager for persistent user preferences."""
import json
from pathlib import Path
from typing import Any, Dict, Optional


class SettingsManager:
    """Manages persistent user settings stored in JSON."""
    
    # Default settings
    DEFAULT_SETTINGS = {
        "language": "en",
        "colormap": "jet",
        "display_splashscreen": True
    }
    
    def __init__(self, settings_dir: Optional[Path] = None):
        """Initialize settings manager.
        
        Args:
            settings_dir: Directory to store settings.json. If None, uses Settings/ in script dir.
        """
        if settings_dir is None:
            settings_dir = Path(__file__).parent / "Settings"
        
        self.settings_dir = Path(settings_dir)
        self.settings_file = self.settings_dir / "settings.json"
        
        # Create settings directory if it doesn't exist
        self.settings_dir.mkdir(exist_ok=True)
        
        # Load settings
        self.settings: Dict[str, Any] = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file.
        
        Returns:
            Settings dictionary, or default settings if file doesn't exist
        """
        if not self.settings_file.exists():
            # Create file with default settings
            self._save_settings_with(self.DEFAULT_SETTINGS.copy())
            return self.DEFAULT_SETTINGS.copy()
        
        try:
            with open(self.settings_file, encoding='utf-8') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged = self.DEFAULT_SETTINGS.copy()
                merged.update(loaded)
                return merged
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings: {e}. Using defaults.")
            return self.DEFAULT_SETTINGS.copy()
    
    def _save_settings_with(self, settings_dict: Dict[str, Any]):
        """Save settings dictionary to JSON file."""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving settings: {e}")
    
    def _save_settings(self):
        """Save current settings to JSON file."""
        self._save_settings_with(self.settings)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a setting value and save to file.
        
        Args:
            key: Setting key
            value: Setting value
        """
        self.settings[key] = value
        self._save_settings()
    
    def get_all(self) -> Dict[str, Any]:
        """Get all settings.
        
        Returns:
            Dictionary of all settings
        """
        return self.settings.copy()
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        self.settings = self.DEFAULT_SETTINGS.copy()
        self._save_settings()


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get or create global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
