import os
import re
import sqlite3

import pandas as pd

battles = os.listdir("/content/drive/MyDrive/Colab Notebooks/Data_2")

def create_table():

    conn = sqlite3.connect("/content/drive/MyDrive/Colab Notebooks/data.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE battles (
            id INTEGER PRIMARY KEY,
            url TEXT,
            winner TEXT,
            challenger_name TEXT,
            challenger_lyrics TEXT,
            defender_name TEXT,
            defender_lyrics TEXT
        )
    """)
    conn.commit()
    conn.close()

    conn = sqlite3.connect("/content/drive/MyDrive/Colab Notebooks/data.db")

def insert_data():
    conn = sqlite3.connect("/content/drive/MyDrive/Colab Notebooks/data.db")
    cursor = conn.cursor()
    for num, battle in enumerate(battles):
        with open("/content/drive/MyDrive/Colab Notebooks/Data_2/" + battle) as f:
            text = f.read()

        print(num)

        pattern_url = re.compile(r"URL:\s*(.*?)\n", re.DOTALL)
        pattern_winner = re.compile(r"Winner:\s*(.*?)\n", re.DOTALL)
        pattern_challenger = re.compile(r"CHALLENGER NAME:\s*(.*?)\n", re.DOTALL)
        pattern_challenger_lyrics = re.compile(r"CHALLENGER LYRICS:\s*(.*?)(?=DEFENDER NAME:|$)", re.DOTALL)
        pattern_defender = re.compile(r"DEFENDER NAME:\s*(.*?)\n", re.DOTALL)
        defender_lyrics_start = text.find("DEFENDER LYRICS:")

        url_search = re.search(pattern_url, text)
        winner_search = re.search(pattern_winner, text)
        challenger_name_search = re.search(pattern_challenger, text)
        challenger_lyrics_search = re.search(pattern_challenger_lyrics, text)
        defender_name_search = re.search(pattern_defender, text)

        url = url_search.group(1).strip() if url_search else ""
        winner = winner_search.group(1).strip() if winner_search else ""
        challenger_name = challenger_name_search.group(1).strip() if challenger_name_search else ""
        challenger_lyrics = challenger_lyrics_search.group(1).strip() if challenger_lyrics_search else ""
        defender_name = defender_name_search.group(1).strip() if defender_name_search else ""
        defender_lyrics = text[defender_lyrics_start + len("DEFENDER LYRICS:"):]

        data = url, winner, challenger_name, challenger_lyrics, defender_name, defender_lyrics

        cursor.execute("INSERT INTO battles (url, winner, challenger_name, challenger_lyrics, defender_name, defender_lyrics) VALUES (?, ?, ?, ?, ?, ?)", data)

    conn.commit()
    conn.close()

def get_data():
    conn = sqlite3.connect('/content/drive/MyDrive/Colab Notebooks/data.db')
    df = pd.read_sql_query("SELECT * FROM battles", conn)
    conn.close()
    return df

if __name__ == "__main__":
    create_table()
    insert_data()
