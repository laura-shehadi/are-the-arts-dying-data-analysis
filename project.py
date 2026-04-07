""" A Project by Laura Emily Shehadi

Are opera and musical theatre companies in Canada experiencing declining attendance, 
performances, or ticket revenue between 2014 and 2024?

This project is inspired by the cultural claim (mentioned by Timothée Chalamet) 
that industries such as opera and ballet are “dying,” this project aims to explpore
the evolution of these arts in terms of attendance, ticket revenue, and number of
performances relative to time. This module contains all functions used to load, 
clean, store, query, analyze, and visualize data on Canadian opera and musical 
theatre companies from Statistics Canada (tables 21-10-0186-01 and 21-10-0187-01).


"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


plot_attendance_over_time(att).savefig('images/attendance.png')
plot_performances_over_time(att).savefig('images/performances.png')
plot_revenue_over_time(rev).savefig('images/revenue.png')
plot_attendance_vs_revenue(joined).savefig('images/attendance_vs_revenue.png')
plot_attendance_comparison(att_raw).savefig('images/comparison.png')

###############################################################################
# Constants
###############################################################################

ATTENDANCE_FILE = 'attendance.csv'
REVENUE_FILE = 'revenue.csv'

# North American Industry Classification System (NAICS) from the datasets
TARGET_NAICS = 'Musical theatre and opera companies [711112]'
THEATRE_NAICS = 'Theatre (except musical) companies [711111]'
DANCE_NAICS = 'Dance companies [711120]'
ALL_ARTS_NAICS  = 'Performing arts companies [7111]'

# Raw column names from Statistics Canada CSVs
RAW_YEAR = 'REF_DATE'
RAW_NAICS = 'North American Industry Classification System (NAICS)'
RAW_ATT_METRIC = 'Performances and attendance, not-for-profit'
RAW_REV_METRIC = 'Detailed sources of revenue, not-for-profit'
RAW_VALUE = 'VALUE'

# Raw metric names for attendance.csv
RAW_TOTAL_PERFORMANCES = 'Total performances'
RAW_TOTAL_ATTENDANCE = 'Total attendance'
RAW_AVG_ATTENDANCE = 'Average attendance per performance'

# Raw metric names for revenue.csv
RAW_PERFORMANCE_REVENUE = 'Performance revenue'
RAW_OTHER_SALES = 'Other sales'
RAW_PUBLIC_SECTOR = 'Public sector'
RAW_PRIVATE_SECTOR = 'Private sector'
RAW_LICENSING = 'Licensing of rights'

# Clean column names
YEAR = 'year'
TOTAL_PERFORMANCES = 'total_performances'
TOTAL_ATTENDANCE = 'total_attendance'
AVG_ATTENDANCE = 'avg_attendance_per_performance'
PERFORMANCE_REVENUE = 'performance_revenue'
OTHER_SALES = 'other_sales'
PUBLIC_SECTOR = 'public_sector'
PRIVATE_SECTOR = 'private_sector'
LICENSING = 'licensing_of_rights'

# Database
DB_PATH  = 'project.db'
ATTENDANCE_TABLE = 'attendance'
REVENUE_TABLE = 'revenue'


###############################################################################
# Part 1: Data Loading
###############################################################################

def load_data(file_path: str) -> pd.DataFrame:
    """Return the data from the Statistics Canada CSV file at file_path
    as a DataFrame.

    >>> df = load_data(ATTENDANCE_FILE)
    >>> isinstance(df, pd.DataFrame)
    True
    """
    return pd.read_csv(file_path)


###############################################################################
# Part 2: Data Cleaning
###############################################################################

def filter_naics(df: pd.DataFrame, naics: str) -> pd.DataFrame:
    """Return a copy of df containing only rows where the NAICS column equals
    naics.

    This is used to isolate data for a specific performing arts category.

    >>> df = load_data(ATTENDANCE_FILE)
    >>> filtered = filter_naics(df, TARGET_NAICS)
    >>> (filtered[RAW_NAICS] == TARGET_NAICS).all()
    True
    """
    return df[df[RAW_NAICS] == naics].copy()


def clean_attendance(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned, wide-format DataFrame of attendance data for musical
    theatre and opera companies.

    Steps performed:
    - Filter to TARGET_NAICS rows only
    - Pivot from long to wide format so each metric becomes its own column
    - Rename columns to clean names
    - Cast year to int

    The returned DataFrame has columns: YEAR, TOTAL_PERFORMANCES,
    TOTAL_ATTENDANCE, and AVG_ATTENDANCE.

    >>> df = load_data(ATTENDANCE_FILE)
    >>> cleaned = clean_attendance(df)
    >>> list(cleaned.columns)
    ['year', 'total_performances', 'total_attendance', 'avg_attendance_per_performance']
    """
    filtered = filter_naics(df, TARGET_NAICS)

    pivoted = filtered.pivot(
        index=RAW_YEAR,
        columns=RAW_ATT_METRIC,
        values=RAW_VALUE
    ).reset_index()
    pivoted.columns.name = None

    cleaned = pivoted.rename(columns={
        RAW_YEAR:                YEAR,
        RAW_TOTAL_PERFORMANCES:  TOTAL_PERFORMANCES,
        RAW_TOTAL_ATTENDANCE:    TOTAL_ATTENDANCE,
        RAW_AVG_ATTENDANCE:      AVG_ATTENDANCE,
    })

    cleaned[YEAR] = cleaned[YEAR].astype(int)
    return cleaned.reset_index(drop=True)


def clean_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned, wide-format DataFrame of revenue data for musical
    theatre and opera companies.

    Steps performed:
    - Filter to TARGET_NAICS rows only
    - Pivot from long to wide format so each metric becomes its own column
    - Multiply all revenue values by 1000 (raw data is in thousands of dollars)
    - Rename columns to clean names
    - Cast year to int

    The returned DataFrame has columns: YEAR, PERFORMANCE_REVENUE, OTHER_SALES,
    PUBLIC_SECTOR, PRIVATE_SECTOR, and LICENSING.

    Note: Some revenue values are missing (NaN) in the raw data for certain
    years. These are kept as NaN rather than filled, since we cannot assume
    a value of zero.

    >>> df = load_data(REVENUE_FILE)
    >>> cleaned = clean_revenue(df)
    >>> list(cleaned.columns)
    ['year', 'performance_revenue', 'other_sales', 'public_sector', 'private_sector', 'licensing_of_rights']
    """
    filtered = filter_naics(df, TARGET_NAICS)

    pivoted = filtered.pivot(
        index=RAW_YEAR,
        columns=RAW_REV_METRIC,
        values=RAW_VALUE
    ).reset_index()
    pivoted.columns.name = None

    revenue_cols = [RAW_PERFORMANCE_REVENUE, RAW_OTHER_SALES,
                    RAW_PUBLIC_SECTOR, RAW_PRIVATE_SECTOR, RAW_LICENSING]
    for col in revenue_cols:
        pivoted[col] = pivoted[col] * 1000

    cleaned = pivoted.rename(columns={
        RAW_YEAR:                YEAR,
        RAW_PERFORMANCE_REVENUE: PERFORMANCE_REVENUE,
        RAW_OTHER_SALES:         OTHER_SALES,
        RAW_PUBLIC_SECTOR:       PUBLIC_SECTOR,
        RAW_PRIVATE_SECTOR:      PRIVATE_SECTOR,
        RAW_LICENSING:           LICENSING,
    })

    cleaned[YEAR] = cleaned[YEAR].astype(int)
    return cleaned.reset_index(drop=True)


###############################################################################
# Part 3: Database Setup
###############################################################################

def create_connection(db_path: str) -> sqlite3.Connection:
    """Return a connection to the SQLite database at db_path. The database
    is created if it does not already exist.
    """
    return sqlite3.connect(db_path)


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the attendance and revenue tables in the database at conn
    if they do not already exist.

    Schema:
        attendance(year INTEGER PRIMARY KEY,
                   total_performances REAL,
                   total_attendance REAL,
                   avg_attendance_per_performance REAL)

        revenue(year INTEGER PRIMARY KEY,
                performance_revenue REAL,
                other_sales REAL,
                public_sector REAL,
                private_sector REAL,
                licensing_of_rights REAL)
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            year                           INTEGER PRIMARY KEY,
            total_performances             REAL,
            total_attendance               REAL,
            avg_attendance_per_performance REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS revenue (
            year                INTEGER PRIMARY KEY,
            performance_revenue REAL,
            other_sales         REAL,
            public_sector       REAL,
            private_sector      REAL,
            licensing_of_rights REAL
        )
    """)
    conn.commit()


def insert_data(conn: sqlite3.Connection,
                df: pd.DataFrame,
                table_name: str) -> None:
    """Insert all rows from df into the table named table_name in conn.
    Existing rows with the same primary key are replaced.
    """
    df.to_sql(table_name, conn, if_exists='replace', index=False)


###############################################################################
# Part 4: Database Queries
###############################################################################

def query_join_attendance_revenue(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a DataFrame produced by joining the attendance and revenue tables
    on year, including all columns from both tables.

    Uses: INNER JOIN
    """
    query = """
        SELECT
            attendance.year,
            attendance.total_performances,
            attendance.total_attendance,
            attendance.avg_attendance_per_performance,
            revenue.performance_revenue,
            revenue.other_sales,
            revenue.public_sector,
            revenue.private_sector,
            revenue.licensing_of_rights
        FROM attendance
        INNER JOIN revenue
        ON attendance.year = revenue.year
        ORDER BY attendance.year
    """
    return pd.read_sql_query(query, conn)


def query_total_revenue_by_year(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a DataFrame with the total revenue from all sources combined,
    grouped by year.

    Uses: GROUP BY, SUM (aggregate function), INNER JOIN
    """
    query = """
        SELECT
            attendance.year,
            SUM(revenue.performance_revenue + revenue.other_sales
                + revenue.public_sector + revenue.private_sector) AS total_revenue
        FROM attendance
        INNER JOIN revenue
        ON attendance.year = revenue.year
        GROUP BY attendance.year
        ORDER BY attendance.year
    """
    return pd.read_sql_query(query, conn)


def query_from_year(conn: sqlite3.Connection, start_year: int) -> pd.DataFrame:
    """Return a DataFrame of attendance and performance revenue data for all
    years greater than or equal to start_year.

    Uses: parameterized query (WHERE year >= ?)

    >>> conn = create_connection(':memory:')
    """
    query = """
        SELECT
            attendance.year,
            attendance.total_performances,
            attendance.total_attendance,
            revenue.performance_revenue
        FROM attendance
        INNER JOIN revenue
        ON attendance.year = revenue.year
        WHERE attendance.year >= ?
        ORDER BY attendance.year
    """
    return pd.read_sql_query(query, conn, params=(start_year,))


###############################################################################
# Part 5: Analysis
###############################################################################

def compute_year_over_year_change(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of df with two added columns: '<column>_change' containing
    the year-over-year absolute change in column, and '<column>_pct_change'
    containing the percentage change rounded to 2 decimal places.

    Precondition: df is sorted by YEAR and contains a column named column.
    """
    result = df.copy()
    result[column + '_change'] = result[column].diff()
    result[column + '_pct_change'] = (result[column].pct_change() * 100).round(2)
    return result


def compute_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Return the correlation coefficient between col1 and col2 in df,
    ignoring rows where either value is NaN.

    A value close to 1 or -1 indicates a strong linear relationship; a value
    close to 0 indicates little to no linear relationship.

    Precondition: df contains columns named col1 and col2.
    """
    return df[[col1, col2]].dropna().corr().iloc[0, 1]


###############################################################################
# Part 6: Visualization
###############################################################################

def plot_attendance_over_time(df: pd.DataFrame) -> plt.Figure:
    """Return a figure with a line chart showing total attendance by year.

    Precondition: df contains columns YEAR and TOTAL_ATTENDANCE.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[YEAR], df[TOTAL_ATTENDANCE], marker='o', color='steelblue',
            linewidth=2, markersize=7)
    ax.set_title('Total Attendance at Musical Theatre & Opera Performances (2014–2024)',
                 fontsize=13)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Attendance')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}'))
    ax.set_xticks(df[YEAR])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig


def plot_performances_over_time(df: pd.DataFrame) -> plt.Figure:
    """Return a figure with a line chart showing number of performances by year.

    Precondition: df contains columns YEAR and TOTAL_PERFORMANCES.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[YEAR], df[TOTAL_PERFORMANCES], marker='s', color='darkorange',
            linewidth=2, markersize=7)
    ax.set_title('Number of Performances per Year (2014–2024)', fontsize=13)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Performances')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}'))
    ax.set_xticks(df[YEAR])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig


def plot_revenue_over_time(df: pd.DataFrame) -> plt.Figure:
    """Return a figure with a line chart showing performance revenue by year.

    Precondition: df contains columns YEAR and PERFORMANCE_REVENUE.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[YEAR], df[PERFORMANCE_REVENUE], marker='^', color='mediumseagreen',
            linewidth=2, markersize=7)
    ax.set_title('Performance Revenue (2014–2024)', fontsize=13)
    ax.set_xlabel('Year')
    ax.set_ylabel('Revenue ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${int(x):,}'))
    ax.set_xticks(df[YEAR])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig


def plot_attendance_vs_revenue(df: pd.DataFrame) -> plt.Figure:
    """Return a figure with side-by-side line charts comparing total attendance
    and performance revenue over time, to examine whether the two trends move
    together.

    Precondition: df contains columns YEAR, TOTAL_ATTENDANCE, and
    PERFORMANCE_REVENUE.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df[YEAR], df[TOTAL_ATTENDANCE], marker='o',
             color='steelblue', linewidth=2)
    ax1.set_title('Total Attendance')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Attendance')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}'))
    ax1.set_xticks(df[YEAR])
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2.plot(df[YEAR], df[PERFORMANCE_REVENUE], marker='^',
             color='mediumseagreen', linewidth=2)
    ax2.set_title('Performance Revenue')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Revenue ($)')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${int(x):,}'))
    ax2.set_xticks(df[YEAR])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle('Attendance vs. Performance Revenue Over Time (2014–2024)',
                 fontsize=13)
    fig.tight_layout()
    return fig


def plot_attendance_comparison(df: pd.DataFrame) -> plt.Figure:
    """Return a figure comparing total attendance across musical theatre & opera,
    theatre, and dance companies over time.

    Precondition: df is the raw (unfiltered) attendance DataFrame, containing
    all NAICS categories.
    """
    categories = {
        'Musical Theatre & Opera': TARGET_NAICS,
        'Theatre (non-musical)':   THEATRE_NAICS,
        'Dance':                   DANCE_NAICS,
    }

    fig, ax = plt.subplots(figsize=(11, 5))

    colors = ['steelblue', 'darkorange', 'mediumpurple']
    for (label, naics), color in zip(categories.items(), colors):
        subset = filter_naics(df, naics)
        subset_att = subset[subset[RAW_ATT_METRIC] == RAW_TOTAL_ATTENDANCE]
        ax.plot(subset_att[RAW_YEAR], subset_att[RAW_VALUE],
                marker='o', label=label, color=color, linewidth=2)

    ax.set_title('Attendance by Performing Arts Category (2014–2024)', fontsize=13)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Attendance')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}'))
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig

###############################################################################
# Main (for quick testing)
###############################################################################

if __name__ == '__main__':
    pass
