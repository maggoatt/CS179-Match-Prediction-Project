"""
A module for downloading historical Premier League match data.

To download the match data into the directory `data/raw`, execute `python3 src/download_data.py`.
To download the match data into the directory `DATA_DIRECTORY`, execute `python3 src/download_data.py [-d DATA_DIRECTORY]`.
"""
from argparse import ArgumentParser
from pathlib import Path

import requests


MATCH_DATA_URLS: dict[str, str] = {
    "25-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "24-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "23-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "22-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "21-22": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "20-21": "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
    "19-20": "https://www.football-data.co.uk/mmz4281/1920/E0.csv",
    "18-19": "https://www.football-data.co.uk/mmz4281/1819/E0.csv",
    "17-18": "https://www.football-data.co.uk/mmz4281/1718/E0.csv",
    "16-17": "https://www.football-data.co.uk/mmz4281/1617/E0.csv",
}


def parse_command_line_arguments() -> Path:
    """
    Extracts the directory for storing historical Premier League match data from command-line arguments.
    
    :return: A Path object to the directory for storing historical Premier League match data.
    """
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory",
                        default = Path("data/raw"),
                        type = Path,
                        help = "Path to the directory where historical Premier League match data will be stored.")
    arguments = parser.parse_args()
    return arguments.directory


def download_data(directory: Path, /) -> None:
    """
    Downloads historical Premier League match data to the specified directory.
    
    :param directory: The directory for storing historical Premier League match data.
    :return: None.
    """
    for season, url in MATCH_DATA_URLS.items():
        response = requests.get(url)
        content = response.content.decode("utf-8-sig")
        with open(directory / f"{season}.csv", "w", encoding = "utf-8") as file:
            file.write(content)


def main() -> None:
    """
    Downloads historical Premier League match data to the specified directory.
    
    :return: None.
    """
    data_directory = parse_command_line_arguments()
    data_directory.mkdir(parents = True, exist_ok = True)
    download_data(data_directory)


if __name__ == "__main__":
    main()
