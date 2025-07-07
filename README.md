# Well Log Analysis Backend

This is a Flask-based backend service for well log analysis. The application provides various APIs for processing, analyzing, and visualizing well log data.

## Features

- Quality Control (QC) Pipeline
- Well Log Data Processing:
  - Null Value Handling
  - Data Smoothing
  - Data Normalization
  - Depth Matching
- Petrophysical Calculations:
  - VSH (Shale Volume) Calculation
  - Porosity Calculation
  - GSA (Gamma Ray, Sonic, Apparent Resistivity) Analysis
- Data Visualization:
  - Log Plots
  - Normalization Plots
  - Porosity Plots
  - GSA Plots
  - Smoothing Plots

## Project Structure

```
backend-pp/
├── app.py              # Main application file with all API endpoints
├── config.py           # Configuration settings
├── data/              
│   ├── wells/          # Contains well log data in CSV format
│   └── depth-matching/ # Contains LAS files for depth matching
├── routes/             # API route handlers
└── services/          # Business logic and data processing services
```

## Prerequisites

- Python 3.x
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MichelPT/dataiku-webapp-backend.git
cd dataiku-webapp-backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Locally

1. Make sure you're in the project directory and your virtual environment is activated

2. Start the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5001`

## API Endpoints

### Data Processing
- `POST /api/run-qc` - Run QC pipeline on well log data
- `POST /api/handle-nulls` - Handle null values in data
- `POST /api/fill-null-marker` - Fill null values within marker ranges
- `POST /api/run-smoothing` - Apply smoothing to well log data
- `POST /api/run-interval-normalization` - Normalize data within intervals

### Well Information
- `GET /api/list-wells` - Get list of available wells
- `POST /api/get-well-columns` - Get columns for specified wells

### Visualization
- `POST /api/get-plot` - Get default log plot
- `POST /api/get-normalization-plot` - Get normalization plot
- `POST /api/get-porosity-plot` - Get porosity plot
- `POST /api/get-gsa-plot` - Get GSA analysis plot
- `POST /api/get-smoothing-plot` - Get smoothing results plot

### Analysis
- `POST /api/run-depth-matching` - Perform depth matching
- `POST /api/run-vsh-calculation` - Calculate shale volume
- `POST /api/run-porosity-calculation` - Calculate porosity
- `POST /api/run-gsa-calculation` - Perform GSA analysis
- `POST /api/trim-data` - Trim well log data

## Data Format

The application expects well log data in CSV format with the following columns:
- DEPTH
- GR (Gamma Ray)
- RT (Resistivity)
- NPHI (Neutron Porosity)
- RHOB (Bulk Density)
- MARKER (for stratigraphic markers)

## Error Handling

All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (missing or invalid parameters)
- 404: Not Found (well or data not found)
- 500: Internal Server Error

Errors are returned in JSON format with an "error" key containing the error message.

## Development

The application uses CORS to allow cross-origin requests, which is particularly useful during development when running the frontend on a different port.

For local development, the server runs in debug mode on port 5001.

## License

[Add your license information here]
