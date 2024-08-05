# %%
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import scipy
import numpy as np
import mpmath

mpmath.mp.dps = 300
import os
from pathlib import Path

CURRENT_WORLD_POP = 8_125_468_789
EQUAL_POPULATION = CURRENT_WORLD_POP / 206
url = "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table"
time_format = "%Y_%m_%d %H_%M_%S"
current_working_dir = Path(os.getcwd())
max_second_delta = 60 * 30

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


def poisson_cdf(k, mu):
    return mpmath.nsum(lambda x: mpmath.exp(-mu) * (mu**x) / mpmath.fac(x), [0, k])


def poisson_ppf(q, mu):
    k = 0
    while poisson_cdf(k, mu) < q:
        k += 1
        if k >= 75:
            return np.nan
    return k


# def binomial_cdf(k, n, p):
#     return mpmath.nsum(
#         lambda x: mpmath.binomial(n, x) * (p**x) * ((1 - p) ** (n - x)), [0, k]
#     )


def calculate_p_values(df: pd.DataFrame, column, result_name, equiv_medal_name):
    # probability person wins medal
    log_result_name = f"log {result_name}"

    p = float(df[column].sum() / CURRENT_WORLD_POP)

    prob_at_least_this_num = lambda row: float(
        1 - poisson_cdf(row[column] - 1, p * row["population"])
    )

    population_adjusted_medals = lambda row: poisson_ppf(
        float(1 - row[result_name]), EQUAL_POPULATION * p
    )

    df[result_name] = df.apply(prob_at_least_this_num, axis=1)
    df[log_result_name] = -np.log(df[result_name])
    df[equiv_medal_name] = df.apply(population_adjusted_medals, axis=1)

    # we still have some precision errors with making equivilent number
    # of medals, we will just linear interpolate with the square of the number medals
    # if you squint at the normal distribution this will make sense particularly as
    # x-mu is large compared to theta,
    equiv_medal_is_nan = df[equiv_medal_name].isna()
    df_no_nan = df.dropna()  # [~equiv_medal_is_nan]
    # y=mx+c
    m, c, r_value, p_value, std_err = scipy.stats.linregress(
        df_no_nan[log_result_name],
        df_no_nan[equiv_medal_name] ** 2,  # equivilent number of medals squared
    )
    # return df.loc[equiv_medal_is_nan]
    df.loc[equiv_medal_is_nan, equiv_medal_name] = (
        (df.loc[equiv_medal_is_nan, log_result_name] * m + c) ** 0.5
    ).astype(int)

    return df


p_gold = "likelihood of this many golds"
equiv_num_gold = "Corrected Gold"
p_total = "likelihood of this many medals"
equiv_num_medals = "Corrected Total Medals"
df = calculate_p_values(df, gold, p_gold, equiv_num_gold)
df = calculate_p_values(df, total, p_total, equiv_num_medals)
df = df.sort_values(by=equiv_num_gold, ascending=False)

df = df[df[country_name] != "Individual Neutral Athletes[A]"]


df[rank] = list(range(len(df)))
df[rank] = df[rank] + 1

df["Golds per 100 Million"] = 1e8 * df["Gold"] / df["population"]
df["Medals per 100 Million"] = 1e8 * df["Total"] / df["population"]


# %%
def format_dataframe(df, advanced_statistics):
    # df[p_gold] = df[p_gold].apply(lambda x: f"{x:.7f}")
    df[equiv_num_medals] = df[equiv_num_medals].astype(int)
    df["population"] = df["population"].astype(int)
    if not advanced_statistics:
        return (
            df[
                [
                    rank,
                    country_name,
                    "population",
                    gold,
                    equiv_num_gold,
                    total,
                    equiv_num_medals,
                ]
            ]
            .rename(columns={total: "Total Medals", "population": "Population"})
            .style.format({p_gold: "{:.9f}", p_total: "{:.9f}"})
        )

    return df.style.format({p_gold: "{:.9f}", p_total: "{:.9f}"})


st.title("The Olympics' most exceptional country (statistically) is...")

best_country = df[country_name].iloc[0]
html_content = f"""
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
    <h1 style="color: #ff5733; font-size: 48px; font-family: 'Arial';">{best_country}</h1>
</div>
"""

# Display the HTML content in Streamlit
st.markdown(html_content, unsafe_allow_html=True)
left_col, right_col = st.columns([4, 1])
with left_col:
    st.markdown(
        """
        The table shows the number of medals a country would win if we **correctly** adjusted for population using statistics!
        """
    )

with right_col:
    # st.markdown("<br/><br/>", unsafe_allow_html=True)
    advanced_stats_dataframe = st.checkbox("Advanced Stats")

st.dataframe(format_dataframe(df, advanced_stats_dataframe))

st.markdown("\* _Host Country_")

st.markdown(
    f"""
    **How did I get these numbers?**

    These numbers were calculated by assuming every person in the world has an equal chance of winning a medal. The top countries had the most statistically significant performance given their population. Interpret the table as "That country is doing well given there population!". This is much more meaningful the medals per Capita.

    For the full methodology and code, check out: [Population Adjusted Medals](https://github.com/lachlan-git/exceptional_olympic_performance)
    """,
    unsafe_allow_html=True,
)
