"""
Internationalization (i18n) support for Quantum Task Planner
Multi-language support with regional compliance features
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class QuantumI18n:
    """Global internationalization manager for quantum task planner"""
    
    def __init__(self, default_locale: str = "en_US"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations: Dict[str, Dict[str, str]] = {}
        self.supported_locales = [
            "en_US",  # English (United States)
            "en_GB",  # English (United Kingdom) 
            "es_ES",  # Spanish (Spain)
            "fr_FR",  # French (France)
            "de_DE",  # German (Germany)
            "ja_JP",  # Japanese (Japan)
            "zh_CN",  # Chinese (Simplified)
            "ko_KR",  # Korean (South Korea)
            "pt_BR",  # Portuguese (Brazil)
            "ru_RU"   # Russian (Russia)
        ]
        
        self._load_translations()
        logger.info(f"QuantumI18n initialized with default locale: {default_locale}")
    
    def _load_translations(self):
        """Load translation files for all supported locales"""
        translations_dir = Path(__file__).parent / "translations"
        
        # Default English translations
        self.translations["en_US"] = {
            "quantum.task.created": "Quantum task created successfully",
            "quantum.task.executed": "Quantum task executed",
            "quantum.task.failed": "Quantum task execution failed",
            "quantum.engine.initialized": "Quantum engine initialized",
            "quantum.coherence.maintained": "Quantum coherence maintained",
            "quantum.coherence.lost": "Quantum coherence lost",
            "quantum.entanglement.created": "Quantum entanglement established",
            "quantum.superposition.collapsed": "Quantum superposition collapsed",
            "dependency.resolved": "Task dependencies resolved",
            "dependency.circular": "Circular dependency detected",
            "performance.excellent": "Excellent performance",
            "performance.good": "Good performance", 
            "performance.poor": "Poor performance - optimization needed",
            "security.validated": "Security validation passed",
            "security.warning": "Security warning detected",
            "deployment.ready": "System ready for production deployment",
            "error.general": "An error occurred",
            "error.validation": "Validation error",
            "error.timeout": "Operation timed out"
        }
        
        # Additional locale translations (simplified for demo)
        locale_translations = {
            "es_ES": {
                "quantum.task.created": "Tarea cuántica creada exitosamente",
                "quantum.task.executed": "Tarea cuántica ejecutada",
                "quantum.engine.initialized": "Motor cuántico inicializado",
                "performance.excellent": "Rendimiento excelente",
                "deployment.ready": "Sistema listo para despliegue en producción"
            },
            "fr_FR": {
                "quantum.task.created": "Tâche quantique créée avec succès",
                "quantum.task.executed": "Tâche quantique exécutée", 
                "quantum.engine.initialized": "Moteur quantique initialisé",
                "performance.excellent": "Performance excellente",
                "deployment.ready": "Système prêt pour le déploiement en production"
            },
            "de_DE": {
                "quantum.task.created": "Quantenaufgabe erfolgreich erstellt",
                "quantum.task.executed": "Quantenaufgabe ausgeführt",
                "quantum.engine.initialized": "Quantenmotor initialisiert", 
                "performance.excellent": "Ausgezeichnete Leistung",
                "deployment.ready": "System bereit für Produktionsbereitstellung"
            },
            "ja_JP": {
                "quantum.task.created": "量子タスクが正常に作成されました",
                "quantum.task.executed": "量子タスクが実行されました",
                "quantum.engine.initialized": "量子エンジンが初期化されました",
                "performance.excellent": "優秀なパフォーマンス",
                "deployment.ready": "システムは本番環境展開準備完了"
            },
            "zh_CN": {
                "quantum.task.created": "量子任务创建成功",
                "quantum.task.executed": "量子任务已执行",
                "quantum.engine.initialized": "量子引擎已初始化",
                "performance.excellent": "性能优异", 
                "deployment.ready": "系统已准备好生产部署"
            }
        }
        
        # Merge translations with fallback to English
        for locale in self.supported_locales:
            if locale not in self.translations:
                self.translations[locale] = {}
            
            # Add specific translations if available
            if locale in locale_translations:
                self.translations[locale].update(locale_translations[locale])
            
            # Fill missing translations with English fallback
            for key, value in self.translations["en_US"].items():
                if key not in self.translations[locale]:
                    self.translations[locale][key] = value
    
    def set_locale(self, locale: str) -> bool:
        """Set current locale for translations"""
        if locale in self.supported_locales:
            self.current_locale = locale
            logger.info(f"Locale changed to: {locale}")
            return True
        else:
            logger.warning(f"Unsupported locale: {locale}")
            return False
    
    def get(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Get translated string for key"""
        target_locale = locale or self.current_locale
        
        # Get translation with fallback chain
        translation = (
            self.translations.get(target_locale, {}).get(key) or
            self.translations.get(self.default_locale, {}).get(key) or
            key  # Fallback to key itself
        )
        
        # Format with kwargs if provided
        try:
            if kwargs:
                translation = translation.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Translation formatting error for key '{key}': {e}")
        
        return translation
    
    def get_supported_locales(self) -> list:
        """Get list of supported locales"""
        return self.supported_locales.copy()
    
    def get_current_locale(self) -> str:
        """Get current active locale"""
        return self.current_locale

# Global instance
_quantum_i18n = QuantumI18n()

def t(key: str, **kwargs) -> str:
    """Convenience function for translations"""
    return _quantum_i18n.get(key, **kwargs)

def set_locale(locale: str) -> bool:
    """Set global locale"""
    return _quantum_i18n.set_locale(locale)

def get_current_locale() -> str:
    """Get current global locale"""
    return _quantum_i18n.get_current_locale()

def get_supported_locales() -> list:
    """Get supported locales"""
    return _quantum_i18n.get_supported_locales()