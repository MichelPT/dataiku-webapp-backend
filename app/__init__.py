# api/app/__init__.py
from flask import Flask
from flask_cors import CORS
from config import DevelopmentConfig

def create_app(config_class=DevelopmentConfig):
    """
    Membuat dan mengkonfigurasi instance aplikasi Flask.
    Pola ini disebut 'Application Factory'.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Inisialisasi ekstensi, seperti CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}}) # Sesuaikan origins untuk produksi

    # Daftarkan semua blueprints (rute) dari aplikasi
    from .routes.qc_routes import qc_bp
    from .routes.plot_routes import plot_bp

    app.register_blueprint(qc_bp, url_prefix='/api/qc')
    app.register_blueprint(plot_bp, url_prefix='/api/plot')

    @app.route('/api/health')
    def health_check():
        return {"status": "ok"}, 200

    return app