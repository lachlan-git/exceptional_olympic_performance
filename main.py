# %%
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import scipy

import os
from pathlib import Path

CURRENT_WORLD_POP = 8_125_468_789

url = "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table"
time_format = "%Y_%m_%d %H_%M_%S"
current_working_dir = Path(os.getcwd())
max_second_delta = 60 * 30

# %%
# %%
# URL of the Wikipedia page
populations = pd.read_csv(current_working_dir / "pop.txt")
populations["population"] = pd.to_numeric(
    populations["population"].str.replace(",", "")
)


def get_updated_dataframe_from_wiki(url):

    # have no idea why the tutorial
    table_class = "wikitable sortable jquery-tablesorter"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    indiatable = soup.find("table", {"class": "wikitable"})
    # Parse the HTML content using BeautifulSoup

    html_table = soup.find("table", {"class": "wikitable"})
    tables_on_page = pd.read_html(str(html_table))
    assert len(tables_on_page) == 1
    return tables_on_page[0]


def get_most_recent_path(
    current_working_dir: Path, time_format: str
) -> tuple[pd.DataFrame, datetime]:

    list_of_paths = sorted(
        [
            (path, datetime.strptime(path.stem, time_format))
            for path in current_working_dir.glob("*.csv")
        ],
        key=lambda path: path[1],
    )
    if len(list_of_paths) == 0:
        return "2024_08_01 18_00_18", datetime.strptime(
            "2024_08_01 18_00_18", time_format
        )

    path, time = list_of_paths[-1]
    return path, time


def get_cached_table(
    url: str, current_working_dir: Path, time_format: str, max_second_delta: int
):
    current_time = datetime.now()
    most_recent_path, most_recent_time = get_most_recent_path(
        current_working_dir, time_format
    )
    if (current_time - most_recent_time).total_seconds() >= max_second_delta:
        most_recent_path = current_working_dir / (
            current_time.strftime(time_format) + ".csv"
        )
        df = get_updated_dataframe_from_wiki(url)
        df.to_csv(most_recent_path, index=False)

    return pd.read_csv(most_recent_path)


df = get_cached_table(url, current_working_dir, time_format, max_second_delta).iloc[:-1]
rank, country_name, gold, silver, bronze, total = list(df.columns)

df = pd.merge(df, populations, how="left", on="NOC")


def calculate_p_values(df: pd.DataFrame, column, result_name):
    # probability person wins medal
    p = float(df[column].sum() / CURRENT_WORLD_POP)
    prob_at_least_this_num = lambda row: 1 - scipy.stats.binom.cdf(
        row[column] - 1, row["population"], p
    )

    df[result_name] = df.apply(prob_at_least_this_num, axis=1)

    return df


p_gold = "P(This Many Golds)"
p_total = "P(This Many Medals)"
df = calculate_p_values(df, gold, p_gold)
df = calculate_p_values(df, total, p_total)
df[rank] = list(range(len(df)))
df[rank] = df[rank] + 1
df = df.sort_values(by=p_gold)


# %%
def format_dataframe(df):
    # df[p_gold] = df[p_gold].apply(lambda x: f"{x:.7f}")
    return df[
        [rank, country_name, p_gold, p_total, gold, silver, bronze, total, "population"]
    ].style.format({p_gold: "{:.9f}", p_total: "{:.9f}"})


st.title("The Olympics' most exceptional country (statistically) is...")

best_country = df[country_name].iloc[0]
html_content = f"""
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
    <h1 style="color: #ff5733; font-size: 48px; font-family: 'Arial';">{best_country}</h1>
</div>
"""

# Display the HTML content in Streamlit
st.markdown(html_content, unsafe_allow_html=True)

st.markdown("and here are the stats...")

st.dataframe(format_dataframe(df))

st.markdown(
    """
    *The table shows the probability of a country achieving this many medals, assuming every citizen has an equal chance of winning, highlighting how statistically improbable a countries performance is.*
    """
)
