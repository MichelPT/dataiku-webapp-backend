# api/config.py
import os

class Config:
    """Konfigurasi dasar."""
    SECRET_KEY = os.environ.get('secret_but_not_private')
    # Tambahkan konfigurasi lain di sini

class DevelopmentConfig(Config):
    """Konfigurasi untuk development."""
    DEBUG = True

class ProductionConfig(Config):
    """Konfigurasi untuk production."""
    DEBUG = False