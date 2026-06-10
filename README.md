# Premier League Match Prediction
Our group will use historical Premier League match data to learn a Bayesian team-strength model for match prediction. The model will learn latent attacking and defensive strengths for each team, use these strengths to predict home and away goal distributions, and use these distributions to predict match outcomes.

## Getting Started
1) Execute `python3 -m venv venv`.
2) Execute `source venv/bin/activate`.
3) Execute `pip install -r requirements.txt`.
4) Execute `python3 -m src.data.download_data` to download historical Premier League match data into the directory `data/raw`. To download the match data into the directory `DATA_DIRECTORY`, execute `python3 -m src.data.download_data -d [DATA_DIRECTORY]`.
5) Execute `python3 -m src.data.data_cleaning` to process the downloaded match data into `data/processed/epl_matches.csv`.
6) Execute `python3 -m src.data.add_team_id` to add team IDs to the processed match data.