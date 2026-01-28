"""Language manager for multi-language GUI support."""
import json
from pathlib import Path
from typing import Dict, Any, Optional


class LanguageManager:
    """Manages language selection and translation lookups."""
    
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "pt-br": "Português (Brasil)"
    }
    
    def __init__(self, default_language: str = "en"):
        """Initialize language manager.
        
        Args:
            default_language: Language code to use by default (e.g., 'en', 'pt-br')
        """
        self.current_language = default_language if default_language in self.SUPPORTED_LANGUAGES else "en"
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.language_path = Path(__file__).parent / "languages"
        
        # Ensure language directory exists
        self.language_path.mkdir(exist_ok=True)
        
        # Load all available translations
        self._load_translations()
    
    def _load_translations(self):
        """Load all translation files."""
        for lang_code in self.SUPPORTED_LANGUAGES.keys():
            self.translations[lang_code] = self._load_language_file(lang_code)
    
    def _load_language_file(self, language_code: str) -> Dict[str, Any]:
        """Load a single language file.
        
        Args:
            language_code: Language code (e.g., 'en', 'pt-br')
            
        Returns:
            Dictionary of translations, or empty dict if file not found
        """
        file_path = self.language_path / f"{language_code}.json"
        
        if not file_path.exists():
            print(f"Warning: Language file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading language file {file_path}: {e}")
            return {}
    
    def set_language(self, language_code: str):
        """Set the current language.
        
        Args:
            language_code: Language code (e.g., 'en', 'pt-br')
        """
        if language_code in self.SUPPORTED_LANGUAGES:
            self.current_language = language_code
        else:
            print(f"Language {language_code} not supported. Using {self.current_language}")
    
    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language
    
    def get_language_name(self, language_code: Optional[str] = None) -> str:
        """Get human-readable language name.
        
        Args:
            language_code: Language code, or None to use current language
            
        Returns:
            Language name
        """
        code = language_code or self.current_language
        return self.SUPPORTED_LANGUAGES.get(code, code)
    
    def translate(self, key: str, default: str = "") -> str:
        """Get translated string for a key.
        
        Args:
            key: Translation key (dot-separated for nested values, e.g., 'ui.buttons.ok')
            default: Default value if translation not found
            
        Returns:
            Translated string or default value
        """
        trans_dict = self.translations.get(self.current_language, {})
        
        # Navigate nested keys
        keys = key.split('.')
        value = trans_dict
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def get_all_translations(self, language_code: Optional[str] = None) -> Dict[str, Any]:
        """Get all translations for a language.
        
        Args:
            language_code: Language code, or None to use current language
            
        Returns:
            Dictionary of all translations
        """
        code = language_code or self.current_language
        return self.translations.get(code, {})
    
    def reload_translations(self):
        """Reload all translation files from disk."""
        self._load_translations()


# Global language manager instance
_language_manager: Optional[LanguageManager] = None


def get_language_manager() -> LanguageManager:
    """Get or create global language manager instance."""
    global _language_manager
    if _language_manager is None:
        _language_manager = LanguageManager()
    return _language_manager


def _(key: str, default: str = "") -> str:
    """Shorthand for translate function.
    
    Args:
        key: Translation key
        default: Default value
        
    Returns:
        Translated string
    """
    return get_language_manager().translate(key, default)
