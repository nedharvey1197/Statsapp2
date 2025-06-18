# Clinical Trials Analysis Dashboard

A web-based dashboard for analyzing clinical trials data, providing both interactive visualizations and exportable PDF reports.

## Features

- Interactive dashboard for data visualization
- PDF report generation
- Statistical analysis of clinical trials data
- Data export capabilities

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Project Structure

- `app.py`: Main application file
- `export_utils.py`: Utilities for PDF export
- `analysis/`: Statistical analysis modules
- `data/`: Data files
- `plots/`: Generated plot files
- `outputs/`: Generated output files

## Requirements

See `requirements.txt` for full list of dependencies. 