from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    # A simple route to confirm the API is running
    return "Flask backend is running!"


if __name__=="__main__":
    app.run()