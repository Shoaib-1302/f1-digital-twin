# src/pages/_imports.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Setup paths
if os.path.exists('/mount/src/f1-digital-twin'):
    BASE_PATH = '/mount/src/f1-digital-twin'
    sys.path.insert(0, f'{BASE_PATH}/src')
elif os.path.exists('/app/f1-digital-twin'):
    BASE_PATH = '/app/f1-digital-twin'
    sys.path.insert(0, f'{BASE_PATH}/src')
else:
    BASE_PATH = str(Path(__file__).parent.parent.parent)
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Try imports
try:
    from utils.visualizations import *
    from utils.helpers import *
    from models.predictor import F1Predictor
    from models.rag_pipeline import F1RAGPipeline
    from data.news_scraper import F1NewsCollector
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    print(f"Import error: {e}")

def get_data_path(filename):
    possible_paths = [
        f'data/{filename}',
        f'{BASE_PATH}/data/{filename}',
        filename
    ]
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None
