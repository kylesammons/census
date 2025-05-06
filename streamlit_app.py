import streamlit as st
import pandas as pd
import re
import requests

# --- Helper Function ---
def clean_header_for_bigquery(header: str) -> str:
    cleaned_header = header.lower()
    cleaned_header = re.sub(r'[^a-z0-9_]', '_', cleaned_header)
    if cleaned_header[0].isdigit():
        cleaned_header = 'col_' + cleaned_header
    cleaned_header = cleaned_header[:128]
    cleaned_header = re.sub(r'_+', '_', cleaned_header)
    cleaned_header = cleaned_header.strip('_')
    return cleaned_header

def remove_unnamed_columns(df):
    return df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]

zip_to_dma = pd.read_csv('Zip Code to DMA - Zipcode Reference.csv')
zip_to_dma['zip_code_tabulation_area'] = zip_to_dma['zip_code_tabulation_area'].astype(str)
dma_df = pd.DataFrame(zip_to_dma)

st.subheader("Census Data API")
# --- Tabs ---
tab1, tab2 = st.tabs(["Fetch from API", "Join Files"])

# --- Tab 1: Census API ---
with tab1:

    acs_table = pd.read_excel('ACS2023_Table_Shells.xlsx')
    filtered_table = acs_table[acs_table['Data Release'].notnull()]
    dropdown_df = filtered_table[['Table ID', 'Stub']].drop_duplicates().reset_index(drop=True)

    st.subheader("Select Table ID")
    selected_row_index = st.selectbox(
        "Select a Table ID and Stub:",
        dropdown_df.index,
        format_func=lambda x: f"{dropdown_df.loc[x, 'Table ID']} - {dropdown_df.loc[x, 'Stub']}"
    )

    selected_row = dropdown_df.iloc[selected_row_index]
    st.dataframe(dropdown_df)

    if st.button("Run"):
        if selected_row.empty:
            st.warning("No rows selected!")
        else:
            table_id = selected_row['Table ID']
            st.success(f"Selected Table ID: {table_id}")
            acs_api_key = '7ce51a1a10c35984ece6b9437d678e834473a20b'
            geo = '&for=zip%20code%20tabulation%20area:*'
            metadata_link = 'https://api.census.gov/data/2023/acs/acs5'
            url = f'''{metadata_link}?get=group({table_id}){geo}&key={acs_api_key}'''
            response = requests.get(url)
            nice = response.json()
            clean_df = pd.DataFrame(nice[1:], columns=nice[0])
            census_df = clean_df
            census_transposed = census_df.T.reset_index()

            acs_table['UniqueID'] = acs_table['UniqueID'].astype(str) + 'E'
            census_transposed.columns = ['UniqueID'] + list(census_transposed.columns[1:])
            census_transposed['UniqueID'] = census_transposed['UniqueID'].astype(str)

            merged_df = pd.merge(census_transposed, acs_table[['UniqueID', 'Stub']], on='UniqueID', how='left')
            merged_df['UniqueID'] = merged_df['Stub'].combine_first(merged_df['UniqueID'])

            new_headers = merged_df['UniqueID'].values
            final_census_df = census_df.copy()
            final_census_df.columns = new_headers

            cleaned_headers = [clean_header_for_bigquery(col) for col in final_census_df.columns]
            final_census_df.columns = cleaned_headers

            final_census_df = final_census_df.loc[:, ~final_census_df.columns.duplicated()]
            df_filtered = final_census_df[[col for col in final_census_df.columns if table_id.lower() not in col]]
            merged_df = pd.merge(df_filtered, dma_df, on='zip_code_tabulation_area', how='outer')
            st.dataframe(merged_df)
            

# --- Tab 2: Join Files ---
with tab2:
    st.subheader("Upload and Join Multiple Census Files")
    uploaded_files = st.file_uploader("Upload multiple CSV files", accept_multiple_files=True, type="csv")

    if uploaded_files and st.button("Join Files"):
        dfs = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            df.columns = [clean_header_for_bigquery(col) for col in df.columns]
            # Ensure zip_code_tabulation_area is string
            if 'zip_code_tabulation_area' in df.columns:
                df['zip_code_tabulation_area'] = df['zip_code_tabulation_area'].astype(str)
            dfs.append(df)

        # Join on 'zip_code_tabulation_area'
        base_df = dfs[0]
        for df in dfs[1:]:
            base_df = pd.merge(base_df, df, on='zip_code_tabulation_area', how='outer', suffixes=('', '_dup'))

        # Drop duplicate columns
        base_df = base_df.loc[:, ~base_df.columns.duplicated()]

        # Merge with DMA dataframe
        merged_with_dma = pd.merge(base_df, dma_df, on='zip_code_tabulation_area', how='left')

        st.success("Files joined and merged with DMA successfully.")
        st.dataframe(merged_with_dma)
