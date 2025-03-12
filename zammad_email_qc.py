import os
import json
import requests
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import language_tool_python
import textstat
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import time
import threading
import sqlite3
import logging
from flask import Flask, request, jsonify
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("email_qc.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app for webhook
app = Flask(__name__)

class ZammadEmailQC:
    def __init__(self, db_path="email_qc.db"):
        self.zammad_url = os.getenv('ZAMMAD_URL')
        self.api_token = os.getenv('ZAMMAD_API_TOKEN')
        self.template_url = os.getenv('TEMPLATE_FIGMA_URL')
        self.headers = {
            'Authorization': f'Token token={self.api_token}',
            'Content-Type': 'application/json'
        }
        
        # Initialize NLP tools
        self.nlp = spacy.load('en_core_web_sm')
        self.sia = SentimentIntensityAnalyzer()
        self.language_tool = language_tool_python.LanguageTool('en-US')
        
        # Initialize database
        self.db_path = db_path
        self.init_database()
        
        # Template patterns (extracted from Figma templates)
        self.template_patterns = self.load_template_patterns()
        
        # Scoring weights
        self.scoring_weights = {
            'spelling_grammar': 0.25,  # 25% of total score
            'tone': 0.20,              # 20% of total score
            'empathy': 0.20,           # 20% of total score
            'template_consistency': 0.20, # 20% of total score
            'response_time': 0.15      # 15% of total score
        }
        
        # Empathy phrases and words
        self.empathy_phrases = [
            "i understand", "i appreciate", "thank you for", "i'm sorry",
            "we apologize", "we understand", "i can see why", "i recognize",
            "this must be", "that sounds", "i can imagine", "we value"
        ]
        
        # Positive tone phrases
        self.positive_phrases = [
            "happy to help", "pleased to", "glad to", "looking forward",
            "thank you", "appreciate", "welcome", "delighted"
        ]
        
        # Negative tone phrases to avoid
        self.negative_phrases = [
            "unfortunately", "cannot", "unable to", "not possible", 
            "you have to", "you must", "you should have", "policy states"
        ]
        
    def init_database(self):
        """Initialize SQLite database for storing QC results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS qc_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER,
            article_id INTEGER,
            agent_id INTEGER,
            timestamp TEXT,
            email_body TEXT,
            spelling_grammar_score REAL,
            tone_score REAL,
            empathy_score REAL,
            template_consistency_score REAL,
            response_time_score REAL,
            total_score REAL,
            feedback TEXT,
            recommendations TEXT,
            FOREIGN KEY (agent_id) REFERENCES agents (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS qc_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qc_result_id INTEGER,
            spelling_errors INTEGER,
            grammar_errors INTEGER,
            readability_score REAL,
            sentiment_score REAL,
            empathy_phrases_count INTEGER,
            response_time_mins REAL,
            FOREIGN KEY (qc_result_id) REFERENCES qc_results (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_template_patterns(self):
        """
        Load template patterns from Figma or local configuration
        In a real implementation, you might parse the Figma API or use a local config
        """
        # This is a simplified example - in production, you'd extract these from Figma
        return {
            'greeting_patterns': [
                r'Dear \w+',
                r'Hello \w+',
                r'Hi \w+'
            ],
            'signature_patterns': [
                r'Best regards,\s*\n\s*[\w\s]+',
                r'Kind regards,\s*\n\s*[\w\s]+',
                r'Thanks,\s*\n\s*[\w\s]+'
            ],
            'formatting_patterns': {
                'paragraph_breaks': r'\n\n',
                'bullet_points': r'•\s',
                'numbered_list': r'\d+\.\s'
            },
            'standard_closings': [
                r"If you have any further questions, please don't hesitate to contact us",
                r"Please let us know if you need any further assistance",
                r"We're here to help if you need anything else"
            ]
        }
    
    def fetch_agent_info(self, agent_id):
        """Fetch agent information from Zammad"""
        url = f"{self.zammad_url}/api/v1/users/{agent_id}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            agent_data = response.json()
            
            # Store agent in database if not exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM agents WHERE id = ?", (agent_id,))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO agents (id, name, email) VALUES (?, ?, ?)",
                    (agent_id, agent_data.get('firstname', '') + ' ' + agent_data.get('lastname', ''), agent_data.get('email', ''))
                )
