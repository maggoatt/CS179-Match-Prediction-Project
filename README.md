# Premier League Match Prediction
Our group will use historical Premier League match data to learn a Bayesian team-strength model for match prediction. The model will learn latent attacking and defensive strengths for each team, use these strengths to predict home and away goal distributions, and use these distributions to predict match outcomes.

## Getting Started
1) Execute `python3 -m venv venv`.
2) Execute `source venv/bin/activate`.
3) Execute `pip install -r requirements.txt`.
4) Execute `python3 src/download_data.py` to download historical Premier League match data into the top-level directory `data`. To download the match data into the directory `DATA_DIRECTORY`, execute `python3 src/download_data.py -d [DATA_DIRECTORY]`.
