# Standard library imports
import datetime
import base64

# Related third-party imports
import streamlit as st
from streamlit_elements import elements
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
import searchconsole
import os
import cohere
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

load_dotenv()
# Initialize Cohere client
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
co = cohere.Client(COHERE_API_KEY)

# Configuration: Set to True if running locally, False if running on Streamlit Cloud
IS_LOCAL = False

# Constants
SEARCH_TYPES = ["web", "image", "video", "news", "discover", "googleNews"]
DATE_RANGE_OPTIONS = [
    "Last 7 Days",
    "Last 30 Days",
    "Last 3 Months",
    "Last 6 Months",
    "Last 12 Months",
    "Last 16 Months",
    "Custom Range"
]
DEVICE_OPTIONS = ["All Devices", "desktop", "mobile", "tablet"]
BASE_DIMENSIONS = ["page", "query", "country", "date"]
MAX_ROWS = 250_000
DF_PREVIEW_ROWS = 100

# -------------
# Streamlit App Configuration
# -------------

def setup_streamlit():
    """
    Configures Streamlit's page settings and displays the app title and markdown information.
    Sets the page layout, title, and markdown content with links and app description.
    """
    st.set_page_config(page_title="✨ Simple Google Search Console Data | LeeFoot.co.uk", layout="wide")
    st.title("✨ Simple Google Search Console Data | June 2024")
    st.markdown(f"### Lightweight GSC Data Extractor. (Max {MAX_ROWS:,} Rows)")

    st.markdown(
        """
        <p>
            Created by <a href="https://twitter.com/LeeFootSEO" target="_blank">LeeFootSEO</a> |
            <a href="https://leefoot.co.uk" target="_blank">More Apps & Scripts on my Website</a>
        """,
        unsafe_allow_html=True
    )
    st.divider()

def init_session_state():
    """
    Initialises or updates the Streamlit session state variables for property selection,
    search type, date range, dimensions, and device type.
    """
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'selected_search_type' not in st.session_state:
        st.session_state.selected_search_type = 'web'
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = 'Last 7 Days'
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=7)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.date.today()
    if 'selected_dimensions' not in st.session_state:
        st.session_state.selected_dimensions = ['page', 'query']
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = 'All Devices'
    if 'custom_start_date' not in st.session_state:
        st.session_state.custom_start_date = datetime.date.today() - datetime.timedelta(days=7)
    if 'custom_end_date' not in st.session_state:
        st.session_state.custom_end_date = datetime.date.today()


def fetch_content(url):
    """
    Fetches the content of a webpage.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(separator=' ', strip=True)
        return content
    except requests.RequestException as e:
        return str(e)
    
def generate_embeddings(text_list):
    """
    Generates embeddings for a list of texts using Cohere's API.
    """
    if not text_list:
        return []

    model = 'embed-english-v3.0'
    input_type = 'search_document'
    response = co.embed(model=model, texts=text_list, input_type=input_type)
    embeddings = response.embeddings
    return embeddings


def calculate_relevancy_scores(df):
    """
    Calculates relevancy scores for each row in the dataframe.
    """
    try:
        st.write("Calculating relevancy scores...")
        st.write(f"Input DataFrame shape: {df.shape}")
        st.write(f"Input DataFrame columns: {df.columns}")
        
        page_contents = [fetch_content(url) for url in df['page']]
        st.write(f"Fetched {len(page_contents)} page contents")
        
        page_embeddings = generate_embeddings(page_contents)
        st.write(f"Generated {len(page_embeddings)} page embeddings")
        
        query_embeddings = generate_embeddings(df['query'].tolist())
        st.write(f"Generated {len(query_embeddings)} query embeddings")
        
        relevancy_scores = cosine_similarity(query_embeddings, page_embeddings).diagonal()
        st.write(f"Calculated {len(relevancy_scores)} relevancy scores")
        st.write(f"Sample relevancy scores: {relevancy_scores[:5]}")
        
        df = df.assign(relevancy_score=relevancy_scores)
        st.write(f"Assigned relevancy scores to DataFrame")
        st.write(f"DataFrame shape after assigning scores: {df.shape}")
        st.write(f"DataFrame columns after assigning scores: {df.columns}")
        st.write(f"Sample relevancy scores from DataFrame: {df['relevancy_score'].head()}")
        
    except Exception as e:
        st.warning(f"Error calculating relevancy scores: {e}")
        df = df.assign(relevancy_score=0)  # Default value if calculation fails
    
    return df
def fetch_data_loading(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    """
    Fetches Google Search Console data with a loading indicator and calculates relevancy scores.
    """
    with st.spinner('Fetching data and calculating relevancy scores...'):
        df = fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type)
        if not df.empty:
            df = calculate_relevancy_scores(df)
        st.write(f"Data fetched. Shape: {df.shape}")
        return df
# -------------

# Google Authentication Functions
# -------------

def load_config():
    """
    Loads the Google API client configuration from Streamlit secrets.
    Returns a dictionary with the client configuration for OAuth.
    """
    client_config = {
    "installed": {
        "client_id": os.environ["CLIENT_ID"],
        "client_secret": os.environ["CLIENT_SECRET"],
        "redirect_uris": [os.environ["REDIRECT_URI"]],
    }}
    return client_config

def init_oauth_flow(client_config):
    """
    Initialises the OAuth flow for Google API authentication using the client configuration.
    Sets the necessary scopes and returns the configured Flow object.
    """
    scopes = ["https://www.googleapis.com/auth/webmasters"]
    return Flow.from_client_config(
        client_config,
        scopes=scopes,
        redirect_uri=client_config["installed"]["redirect_uris"][0],
    )

def google_auth(client_config):
    """
    Starts the Google authentication process using OAuth.
    Generates and returns the OAuth flow and the authentication URL.
    """
    flow = init_oauth_flow(client_config)
    auth_url, _ = flow.authorization_url(prompt="consent")
    return flow, auth_url

def auth_search_console(client_config, credentials):
    """
    Authenticates the user with the Google Search Console API using provided credentials.
    Returns an authenticated searchconsole client.
    """
    token = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
        "id_token": getattr(credentials, "id_token", None),
    }
    return searchconsole.authenticate(client_config=client_config, credentials=token)

# -------------
# Data Fetching Functions
# -------------

def list_gsc_properties(credentials):
    """
    Lists all Google Search Console properties accessible with the given credentials.
    Returns a list of property URLs or a message if no properties are found.
    """
    service = build('webmasters', 'v3', credentials=credentials)
    site_list = service.sites().list().execute()
    return [site['siteUrl'] for site in site_list.get('siteEntry', [])] or ["No properties found"]

def fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    """
    Fetches Google Search Console data for a specified property, date range, dimensions, and device type.
    Handles errors and returns the data as a DataFrame.
    """
    query = webproperty.query.range(start_date, end_date).search_type(search_type).dimension(*dimensions)

    if 'device' in dimensions and device_type and device_type != 'All Devices':
        query = query.filter('device', 'equals', device_type.lower())

    try:
        df = query.limit(MAX_ROWS).get().to_dataframe()
        return process_gsc_data(df)
    except Exception as e:
        show_error(e)
        return pd.DataFrame()

def process_gsc_data(df):
    """
    Processes the GSC data to return only unique pages with their first query and relevancy score.
    """
    st.write("Processing GSC data...")
    st.write(f"Input DataFrame shape: {df.shape}")
    st.write(f"Input DataFrame columns: {df.columns}")
    
    # Sort the dataframe by page and clicks (descending) to get the most relevant query first
    df_sorted = df.sort_values(['page', 'clicks'], ascending=[True, False])
    
    # Get the first occurrence of each page (which will be the one with the highest clicks)
    df_unique = df_sorted.drop_duplicates(subset='page', keep='first').copy()
    
    st.write(f"Unique pages DataFrame shape: {df_unique.shape}")
    st.write(f"Unique pages DataFrame columns: {df_unique.columns}")
    
    # Ensure 'relevancy_score' column exists and is preserved
    if 'relevancy_score' not in df_unique.columns:
        st.write("Relevancy score column not found, adding default values")
        df_unique['relevancy_score'] = 0  # Default value if column doesn't exist
    else:
        st.write("Preserving relevancy scores")
        # Make sure to keep the original relevancy scores
        df_unique['relevancy_score'] = df_sorted.groupby('page')['relevancy_score'].first().values
    
    # Select only the relevant columns, including the relevancy_score
    result = df_unique[['page', 'query', 'clicks', 'impressions', 'ctr', 'position', 'relevancy_score']]
    
    st.write(f"Processed data. Shape: {result.shape}")
    st.write(f"Columns: {result.columns}")
    st.write(f"Sample relevancy scores: {result['relevancy_score'].head()}")
    
    return result


def fetch_data_loading(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    """
    Fetches Google Search Console data with a loading indicator and calculates relevancy scores.
    """
    with st.spinner('Fetching data and calculating relevancy scores...'):
        df = fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type)
        st.write(f"Data fetched. Shape: {df.shape}")
        st.write(f"Columns: {df.columns}")
        
        if not df.empty:
            df = calculate_relevancy_scores(df)
            st.write("Relevancy scores calculated.")
            st.write(f"DataFrame shape after calculating scores: {df.shape}")
            st.write(f"DataFrame columns after calculating scores: {df.columns}")
            st.write(f"Sample relevancy scores after calculation: {df['relevancy_score'].head()}")
        
        processed_df = process_gsc_data(df)
        st.write("Data processed.")
        st.write(f"Final DataFrame shape: {processed_df.shape}")
        st.write(f"Final DataFrame columns: {processed_df.columns}")
        st.write(f"Final sample relevancy scores: {processed_df['relevancy_score'].head()}")
        
        return processed_df
    """
    Fetches Google Search Console data with a loading indicator. Utilises 'fetch_gsc_data' for data retrieval.
    Returns the fetched data as a DataFrame.
    """
    with st.spinner('Fetching data...'):
        return fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type)

# -------------
# Utility Functions
# -------------

def update_dimensions(selected_search_type):
    """
    Updates and returns the list of dimensions based on the selected search type.
    Adds 'device' to dimensions if the search type requires it.
    """
    return BASE_DIMENSIONS + ['device'] if selected_search_type in SEARCH_TYPES else BASE_DIMENSIONS

def calc_date_range(selection, custom_start=None, custom_end=None):
    """
    Calculates the date range based on the selected range option.
    Returns the start and end dates for the specified range.
    """
    range_map = {
        'Last 7 Days': 7,
        'Last 30 Days': 30,
        'Last 3 Months': 90,
        'Last 6 Months': 180,
        'Last 12 Months': 365,
        'Last 16 Months': 480
    }
    today = datetime.date.today()
    if selection == 'Custom Range':
        if custom_start and custom_end:
            return custom_start, custom_end
        else:
            return today - datetime.timedelta(days=7), today
    return today - datetime.timedelta(days=range_map.get(selection, 0)), today

def show_error(e):
    """
    Displays an error message in the Streamlit app.
    Formats and shows the provided error 'e'.
    """
    st.error(f"An error occurred: {e}")

def property_change():
    """
    Updates the 'selected_property' in the Streamlit session state.
    Triggered on change of the property selection.
    """
    st.session_state.selected_property = st.session_state['selected_property_selector']

# -------------
# File & Download Operations
# -------------

def show_dataframe(report):
    """
    Shows a preview of the first 100 rows of the processed report DataFrame in an expandable section.
    """
    with st.expander("Preview the First 100 Rows (Unique Pages with Top Query)"):
        st.dataframe(report.head(DF_PREVIEW_ROWS))

def download_csv_link(report):
    """
    Generates and displays a download link for the report DataFrame in CSV format.
    """
    def to_csv(df):
        return df.to_csv(index=False, encoding='utf-8-sig')

    csv = to_csv(report)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="search_console_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# -------------
# Streamlit UI Components
# -------------

def show_google_sign_in(auth_url):
    """
    Displays the Google sign-in button and authentication URL in the Streamlit sidebar.
    """
    with st.sidebar:
        if st.button("Sign in with Google"):
            # Open the authentication URL
            st.write('Please click the link below to sign in:')
            st.markdown(f'[Google Sign-In]({auth_url})', unsafe_allow_html=True)

def show_property_selector(properties, account):
    """
    Displays a dropdown selector for Google Search Console properties.
    Returns the selected property's webproperty object.
    """
    selected_property = st.selectbox(
        "Select a Search Console Property:",
        properties,
        index=properties.index(
            st.session_state.selected_property) if st.session_state.selected_property in properties else 0,
        key='selected_property_selector',
        on_change=property_change
    )
    return account[selected_property]

def show_search_type_selector():
    """
    Displays a dropdown selector for choosing the search type.
    Returns the selected search type.
    """
    return st.selectbox(
        "Select Search Type:",
        SEARCH_TYPES,
        index=SEARCH_TYPES.index(st.session_state.selected_search_type),
        key='search_type_selector'
    )

def show_date_range_selector():
    """
    Displays a dropdown selector for choosing the date range.
    Returns the selected date range option.
    """
    return st.selectbox(
        "Select Date Range:",
        DATE_RANGE_OPTIONS,
        index=DATE_RANGE_OPTIONS.index(st.session_state.selected_date_range),
        key='date_range_selector'
    )

def show_custom_date_inputs():
    """
    Displays date input fields for custom date range selection.
    Updates session state with the selected dates.
    """
    st.session_state.custom_start_date = st.date_input("Start Date", st.session_state.custom_start_date)
    st.session_state.custom_end_date = st.date_input("End Date", st.session_state.custom_end_date)

def show_dimensions_selector(search_type):
    """
    Displays a multi-select box for choosing dimensions based on the selected search type.
    Returns the selected dimensions.
    """
    available_dimensions = update_dimensions(search_type)
    return st.multiselect(
        "Select Dimensions:",
        available_dimensions,
        default=st.session_state.selected_dimensions,
        key='dimensions_selector'
    )

def show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions):
    """
    Displays a button to fetch data based on selected parameters.
    Shows the report DataFrame and download link upon successful data fetching.
    """
    if st.button("Fetch Data"):
        report = fetch_data_loading(webproperty, search_type, start_date, end_date, selected_dimensions)

        if report is not None and not report.empty:
            show_dataframe(report)
            download_csv_link(report)
        else:
            st.warning("No data found for the selected criteria.")



# -------------
# Main Streamlit App Function
# -------------

# Main Streamlit App Function
def main():
    """
    The main function for the Streamlit application.
    Handles the app setup, authentication, UI components, and data fetching logic.
    """
    setup_streamlit()
    client_config = load_config()
    st.session_state.auth_flow, st.session_state.auth_url = google_auth(client_config)

    query_params = st.experimental_get_query_params()
    auth_code = query_params.get("code", [None])[0]

    if auth_code and not st.session_state.get('credentials'):
        st.session_state.auth_flow.fetch_token(code=auth_code)
        st.session_state.credentials = st.session_state.auth_flow.credentials

    if not st.session_state.get('credentials'):
        show_google_sign_in(st.session_state.auth_url)
    else:
        init_session_state()
        account = auth_search_console(client_config, st.session_state.credentials)
        properties = list_gsc_properties(st.session_state.credentials)

        if properties:
            webproperty = show_property_selector(properties, account)
            search_type = show_search_type_selector()
            date_range_selection = show_date_range_selector()

            if date_range_selection == 'Custom Range':
                show_custom_date_inputs()
                start_date, end_date = st.session_state.custom_start_date, st.session_state.custom_end_date
            else:
                start_date, end_date = calc_date_range(date_range_selection)

            selected_dimensions = show_dimensions_selector(search_type)
            
            if st.button("Fetch Data and Calculate Relevancy"):
                report = fetch_data_loading(webproperty, search_type, start_date, end_date, selected_dimensions)

                if report is not None and not report.empty:
                    show_dataframe(report)
                    download_csv_link(report)
                else:
                    st.warning("No data found for the selected criteria.")

if __name__ == "__main__":
    main()