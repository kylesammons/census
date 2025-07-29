import streamlit as st
import pandas as pd
import re
import requests
import zipfile
import io
import pickle
import os

# --- Helper Function ---
def clean_header_for_bigquery(header: str) -> str:
    cleaned_header = header.lower()
    cleaned_header = re.sub(r'[^a-z0-9_]', '_', cleaned_header)
    cleaned_header = cleaned_header[:128]
    cleaned_header = re.sub(r'_+', '_', cleaned_header)
    cleaned_header = cleaned_header.strip('_')
    return cleaned_header

def remove_unnamed_columns(df):
    return df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]

def load_user_data():
    """Load user notes and tags from pickle file"""
    try:
        if os.path.exists('user_table_data.pkl'):
            with open('user_table_data.pkl', 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return {}

def save_user_data(data):
    """Save user notes and tags to pickle file"""
    try:
        with open('user_table_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        return True
    except:
        return False

zip_to_dma = pd.read_excel('Zip Code to DMA.xlsx', dtype={'zip_code_tabulation_area': str})
zip_to_dma['zip_code_tabulation_area'] = zip_to_dma['zip_code_tabulation_area'].str.zfill(5)
for col in zip_to_dma.select_dtypes(include=['object']).columns:
    zip_to_dma[col] = zip_to_dma[col].astype(str).str.strip()
zip_to_dma['DMA'] = zip_to_dma['DMA'].replace({
    'CLEVLAND AKRON CANTON, OHIO': 'CLEVELAND AKRON CANTON, OHIO'
})
dma_df = pd.DataFrame(zip_to_dma)

def categorize_table(stub):
    stub_lower = str(stub).lower()
    if any(word in stub_lower for word in ['race', 'ethnicity', 'hispanic']):
        return "Race and Ethnicity"
    elif any(word in stub_lower for word in ['income', 'poverty', 'earnings']):
        return "Income and Poverty"
    elif any(word in stub_lower for word in ['population', 'age', 'sex']):
        return "Populations and People"
    elif any(word in stub_lower for word in ['housing', 'units', 'rooms', 'rent', 'tenure', 'structure', 'cost', 'price', 'value', 'house', 'vacancy', 'plumbing']):
        return "Housing"
    elif any(word in stub_lower for word in ['disability', 'insurance', 'health']):
        return "Health"
    elif any(word in stub_lower for word in ['citizenship', 'veteran', 'military', 'service']):
        return "Government"
    elif any(word in stub_lower for word in ['family', 'household', 'married', 'children']):
        return "Families and Living Arrangements"
    elif any(word in stub_lower for word in ['school', 'education', 'enrollment', 'degree']):
        return "Education"
    elif any(word in stub_lower for word in ['employment', 'industry', 'occupation', 'work']):
        return "Business and Economy"
    else:
        return "Other"

def create_zip_file(tables_dict):
    """Create a zip file containing all CSV files"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for table_id, (df, stub) in tables_dict.items():
            csv_data = df.to_csv(index=False)
            # Clean stub for filename (remove special characters)
            clean_stub = re.sub(r'[^\w\s-]', '', stub).strip()
            clean_stub = re.sub(r'[-\s]+', '_', clean_stub)
            filename = f"{table_id} - {clean_stub}.csv"
            zip_file.writestr(filename, csv_data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Set Streamlit page config
st.set_page_config(page_title="CensusLAB", page_icon="📊",layout="wide", initial_sidebar_state="expanded")

# Session state tracking
if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = set()
if "fetch_status" not in st.session_state:
    st.session_state.fetch_status = {}
if "fetched_tables" not in st.session_state:
    st.session_state.fetched_tables = {}
if "table_stubs" not in st.session_state:
    st.session_state.table_stubs = {}
if "last_edited_df" not in st.session_state:
    st.session_state.last_edited_df = None
if "previous_topic" not in st.session_state:
    st.session_state.previous_topic = "All"
if "user_data" not in st.session_state:
    st.session_state.user_data = load_user_data()
if "temp_selections" not in st.session_state:  # Track temporary selections
    st.session_state.temp_selections = set()

st.title("CensusLAB")

# ACS
# Load and prepare your dataframe
acs_table = pd.read_excel("ACS2023_Table_Shells.xlsx")
acs_table["Topic"] = acs_table["Stub"].apply(categorize_table)

def update_hierarchical_labels(df, column_name='Stub'):
    """
    Update dataframe to propagate category labels (rows with ':') to subsequent rows
    and add 'All' to category rows. Stops propagation when encountering another ':'
    or a blank/null field.
    
    Args:
        df: pandas DataFrame
        column_name: name of the column to process (default 'Stub')
    
    Returns:
        DataFrame with updated labels
    """
    # Create a copy to avoid modifying the original
    df_updated = df.copy()
    
    current_category = None
    
    for idx in df_updated.index:
        stub_value = df_updated.loc[idx, column_name]
        
        # Check if the field is blank/null
        if pd.isna(stub_value) or str(stub_value).strip() == '' or str(stub_value).strip() == 'nan':
            # Reset category when encountering blank field
            current_category = None
            continue
        
        stub_value = str(stub_value).strip()
        
        # Check if this row contains a ':'
        if ':' in stub_value:
            # This is a category row - add 'All' to it
            current_category = stub_value
            df_updated.loc[idx, column_name] = f"{current_category} All"
        else:
            # This is a subcategory row - prepend the current category if we have one
            if current_category:
                df_updated.loc[idx, column_name] = f"{current_category} {stub_value}"
    
    return df_updated

# Apply the function to update your dataframe in place
acs_table = update_hierarchical_labels(acs_table, 'Stub')

# Cleaning ACS Table
filtered_table = acs_table[acs_table["Data Release"].notnull()]
filtered_table = filtered_table[filtered_table["Table ID"].str.match(r'^[A-Z]\d{5}$', na=False)]
filtered_table = filtered_table.drop_duplicates(subset=["Table ID"])

# Sidebar: Logo and Topic Filter
with st.sidebar:
    st.image("Waves-Logo_Color.svg", width=200)
    
    # Topic filter with radio
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Topic")
    all_topics = sorted(filtered_table["Topic"].dropna().unique())
    selected_topic = st.radio("Topic",["All"] + all_topics, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.84, 0.16])
    with col1:
        st.markdown("### Queue")
    with col2:
        if st.button("", type="secondary", icon="🗑️", help="Clear Queue"):
            st.session_state.selected_ids.clear()
            st.session_state.fetch_status.clear()
            st.session_state.fetched_tables.clear()
            st.session_state.table_stubs.clear()
            st.rerun()

# Update previous topic
st.session_state.previous_topic = selected_topic

# Filter by selected topic
if selected_topic != "All":
    display_table = filtered_table[filtered_table["Topic"] == selected_topic]
else:
    display_table = filtered_table

# Build table to display - no longer sync with session state for Selected column
table_df = display_table[["Table ID", "Stub", "Topic"]].drop_duplicates().reset_index(drop=True)
# Initialize Selected column as False for all rows
table_df["Selected"] = False

# Add Notes and Tag columns from user data
table_df["Notes"] = table_df["Table ID"].apply(lambda x: st.session_state.user_data.get(x, {}).get("notes", ""))
table_df["Tag"] = table_df["Table ID"].apply(lambda x: st.session_state.user_data.get(x, {}).get("tag", ""))

# Build queue DataFrame from ALL selected items
all_table_df = filtered_table[["Table ID", "Stub", "Topic"]].drop_duplicates().reset_index(drop=True)

# Placeholder for sidebar queue
sidebar_queue_placeholder = st.sidebar.empty()

# Build and display the queue in sidebar
queue_df = all_table_df[all_table_df["Table ID"].isin(st.session_state.selected_ids)]

with sidebar_queue_placeholder.container():
    with st.container():
        if queue_df.empty:
            st.markdown("*No tables selected*")
        else:
            for i, row in queue_df.iterrows():
                table_id = row["Table ID"]
                col1, col2, col3 = st.columns([1.5, 3, 1])
                status = st.session_state.fetch_status.get(table_id, "pending")
                icon = {
                    "pending": "🟰",
                    "waiting": "⏳", 
                    "running": "🔄",
                    "done": "✅",
                    "error": "❌"
                }.get(status, "")
                with col1:
                    st.markdown(f"**{table_id}**")
                with col2:
                    st.markdown(row["Stub"][:25] + "..." if len(row["Stub"]) > 25 else row["Stub"])
                with col3:
                    st.markdown(f"{icon}")
        
        # Run all button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("🚀 Run Queue", type="primary", use_container_width=True):
                if not st.session_state.selected_ids:
                    st.warning("No table IDs selected!")
                else:
                    pending_count = sum(1 for status in st.session_state.fetch_status.values() if status == "pending")
                    if pending_count == 0:
                        st.info("All tables in queue have already been processed!")
                    else:
                        for table_id in st.session_state.selected_ids:
                            if st.session_state.fetch_status.get(table_id) == "pending":
                                st.session_state.fetch_status[table_id] = "running"
                                break  # Only start one at a time
                        st.rerun()

# --- Tabs ---
tab1, tab2 = st.tabs(["Library", "Join"])

with tab1:
    st.info(
    """
    📊  CensusLAB helps enrich your data and empowers your team to perform data-driven targeting.

    Filter tables by topic, select relevant data using checkboxes, and add them to your queue. The results are provided at the zip code level with County, State, and DMA fields.
    """
)

    # Column configuration for the data editor
    column_config = {
        "Selected": st.column_config.CheckboxColumn(required=False),
        "Table ID": st.column_config.TextColumn(disabled=True),
        "Stub": st.column_config.TextColumn(disabled=True),
        "Topic": st.column_config.TextColumn(disabled=True),
        "Notes": st.column_config.TextColumn(help="Add your notes about this table"),
        "Tag": st.column_config.SelectboxColumn(
            "Tag",
            help="Tag this table",
            options=["", "⭐ Favorite", "❌ Errors"],
            required=False
        )
    }

    # Editable table with disabled option to prevent auto-rerun on checkbox changes
    edited_df = st.data_editor(
        table_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key=f"data_editor_{selected_topic}",
        disabled=False  # Keep enabled but we handle the rerun differently
    )

    # Count selected tables
    selected_count = len(edited_df[edited_df["Selected"] == True])

    # Create columns for Save button and Add to Queue section
    col1, col2, col3, col4 = st.columns([2, 4, 1, 2])
    
    with col1:
        if st.button("💾 Save Notes ", type="secondary"):
            # Update user_data with notes and tags from edited_df
            for _, row in edited_df.iterrows():
                table_id = row["Table ID"]
                if table_id not in st.session_state.user_data:
                    st.session_state.user_data[table_id] = {}
                st.session_state.user_data[table_id]["notes"] = row["Notes"]
                st.session_state.user_data[table_id]["tag"] = row["Tag"]
            
            # Save to file
            if save_user_data(st.session_state.user_data):
                st.toast("Notes and tags saved successfully!", icon="✅")
            else:
                st.toast("Error saving notes and tags", icon="❌")
    
    with col4:
            # Add to Queue button moved to the furthest right
            button_text = f"➕ Add {selected_count} Table{'s' if selected_count != 1 else ''} to Queue" if selected_count > 0 else "➕ Add to Queue"
            add_to_queue_button = st.button(button_text, type="primary", disabled=(selected_count == 0))
            
            if add_to_queue_button:
                if selected_count > 0:
                    newly_selected_tables = edited_df[edited_df["Selected"] == True]
                    added_count = 0
                    
                    for _, row in newly_selected_tables.iterrows():
                        table_id = row["Table ID"]
                        if table_id not in st.session_state.selected_ids:
                            st.session_state.selected_ids.add(table_id)
                            st.session_state.fetch_status[table_id] = "pending"
                            # Store the stub for this table
                            st.session_state.table_stubs[table_id] = row["Stub"]
                            added_count += 1
                    
                    if added_count > 0:
                        st.toast(f"Added {added_count} table{'s' if added_count != 1 else ''} to queue!", icon="✅")
                        # Reset selections in the table
                        st.rerun()
                    else:
                        st.toast("All selected tables are already in the queue!", icon="ℹ️")

    # Handle fetching for 1 "running" table at a time
    for table_id, status in st.session_state.fetch_status.items():
        if status == "running":
            # Get the stub for this table
            table_stub = st.session_state.table_stubs.get(table_id, "Unknown")
            with st.status(f"Fetching {table_id} - {table_stub}...", expanded=True):
                st.write("Calling Census API...")
                try:
                    acs_api_key = '7ce51a1a10c35984ece6b9437d678e834473a20b'
                    geo = '&for=zip%20code%20tabulation%20area:*'
                    metadata_link = 'https://api.census.gov/data/2023/acs/acs5'
                    url = f"{metadata_link}?get=group({table_id}){geo}&key={acs_api_key}"
                    response = requests.get(url)

                    if response.status_code == 200:
                        st.write("Processing data...")
                        nice = response.json()
                        clean_df = pd.DataFrame(nice[1:], columns=nice[0])
                        census_df = clean_df
                        census_transposed = census_df.T.reset_index()

                        acs_table_copy = acs_table.copy()
                        acs_table_copy['UniqueID'] = acs_table_copy['UniqueID'].astype(str) + 'E'

                        # Update Stub values where Line = 1 in the ACS table copy
                        acs_table_copy = acs_table_copy.reset_index(drop=True)
                        for i in range(1, len(acs_table_copy)):
                            if acs_table_copy.iloc[i]['Line'] == 1:
                                acs_table_copy.iloc[i, acs_table_copy.columns.get_loc('Stub')] = acs_table_copy.iloc[i-1]['Stub']

                        # Remove 'Universe: ' prefix from Stub values
                        acs_table_copy['Stub'] = acs_table_copy['Stub'].str.replace(r'^Universe:\s*', '', regex=True)
                        
                        census_transposed.columns = ['UniqueID'] + list(census_transposed.columns[1:])
                        census_transposed['UniqueID'] = census_transposed['UniqueID'].astype(str)

                        merged_df = pd.merge(
                            census_transposed,
                            acs_table_copy[['UniqueID', 'Stub']],
                            on='UniqueID',
                            how='left'
                        )
                        merged_df['UniqueID'] = merged_df['Stub'].combine_first(merged_df['UniqueID'])

                        new_headers = merged_df['UniqueID'].values
                        final_census_df = census_df.copy()
                        final_census_df.columns = new_headers
                        final_census_df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in final_census_df.columns]
                        final_census_df = final_census_df.loc[:, ~final_census_df.columns.duplicated()]
                        df_filtered = final_census_df[
                            [col for col in final_census_df.columns if table_id.lower() not in col]
                        ]
                        
                        # Drop 'name' and 'geo_id' columns if they exist
                        columns_to_drop = ['name', 'geo_id']
                        df_filtered = df_filtered.drop(columns=[col for col in columns_to_drop if col in df_filtered.columns])
                        
                        # Convert string columns to numeric first, then handle negatives
                        for col in df_filtered.columns:
                            if col not in ['zip_code_tabulation_area', 'state']:  # Skip non-numeric identifier columns
                                # Convert to numeric, keeping non-convertible as NaN
                                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                                # Replace negative values with NaN
                                df_filtered[col] = df_filtered[col].mask(df_filtered[col] < 0, pd.NA)

                        # Merge with zip code reference data (DMA, city, state, county)
                        if 'zip_code_tabulation_area' in df_filtered.columns:
                            # Ensure zip_code_tabulation_area is string for proper merging
                            df_filtered['zip_code_tabulation_area'] = df_filtered['zip_code_tabulation_area'].astype(str).str.zfill(5)
                            df_filtered = pd.merge(df_filtered, dma_df, on='zip_code_tabulation_area', how='left')

                        # Save results with stub
                        st.session_state.fetched_tables[table_id] = (df_filtered, table_stub)
                        st.session_state.fetch_status[table_id] = "done"
                        st.success(f"✅ {table_id} - {table_stub} completed successfully!")
                    else:
                        st.session_state.fetch_status[table_id] = "error"
                        st.error(f"❌ Failed to fetch {table_id} - {table_stub}: HTTP {response.status_code}")
                except Exception as e:
                    st.session_state.fetch_status[table_id] = "error"
                    st.error(f"❌ Error fetching {table_id} - {table_stub}: {str(e)}")

            # Continue with next pending table
            remaining_pending = [tid for tid, status in st.session_state.fetch_status.items() if status == "pending"]
            if remaining_pending:
                st.session_state.fetch_status[remaining_pending[0]] = "running"
            
            st.rerun()
            break  # Ensure only one runs at a time

    # Display fetched tables
    if st.session_state.fetched_tables:
        st.markdown("---")
        st.markdown("#### 📊 Results")
        st.info(
    """
    ⬇️  Preview and download your tables individually or download all into a zip file to begin your targeting/analysis.

    If you'd like to join your downloaded files into one unified table, navigate to the Join tab and follow the prompts.
    """
)

        
        for table_id, (df, stub) in st.session_state.fetched_tables.items():
            with st.expander(f"📋 {table_id} - {stub} ({df.shape[0]:,} rows, {df.shape[1]} columns)", expanded=False):
                # Preview the first 10 rows
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df,use_container_width=True)

                # Convert to CSV for download
                csv = df.to_csv(index=False).encode("utf-8")
                # Clean stub for filename (remove special characters)
                clean_stub = re.sub(r'[^\w\s-]', '', stub).strip()
                clean_stub = re.sub(r'[-\s]+', '_', clean_stub)
                filename = f"{table_id} - {clean_stub}.csv"

                # Download button
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{table_id}"
                )
        
        # Download All Files button
        st.markdown("<br>", unsafe_allow_html=True)
        if len(st.session_state.fetched_tables) > 1:
            zip_data = create_zip_file(st.session_state.fetched_tables)
            st.download_button(
                label="📦 Download All Files Zip",
                data=zip_data,
                file_name="census_tables_download.zip",
                mime="application/zip",
                type="primary",
                help="Download all files in a zip"
            )

# --- Tab 2: Join Files ---
with tab2:
    st.subheader("Upload and Join Multiple Census Files")
    uploaded_files = st.file_uploader("Upload multiple CSV files", accept_multiple_files=True, type="csv")
    
    if uploaded_files and st.button("Join Files"):
        dfs = []
        
        # Columns to exclude from table ID prefixing
        excluded_columns = ['zip_code_tabulation_area', 'State', 'County', 'City', 'DMA']
        
        for file in uploaded_files:
            df = pd.read_csv(file)
            
            # Extract table ID from filename (everything before the first " - ")
            filename = file.name
            if " - " in filename:
                table_id = filename.split(" - ")[0].strip()
            else:
                # Fallback: use filename without extension if no " - " found
                table_id = filename.split(".")[0].strip()
            
            # Apply table ID prefix to columns (except excluded ones)
            new_columns = []
            for col in df.columns:
                if col in excluded_columns:
                    new_columns.append(col)
                else:
                    # Use double underscore as separator, preserving original colons
                    new_columns.append(f"{table_id}__{col}")
            
            df.columns = new_columns
            
            # Ensure zip_code_tabulation_area is string
            if 'zip_code_tabulation_area' in df.columns:
                df['zip_code_tabulation_area'] = df['zip_code_tabulation_area'].astype(str)
                df['zip_code_tabulation_area'] = df['zip_code_tabulation_area'].str.zfill(5)
            
            dfs.append(df)
        
        # Join all dataframes on common geographic columns
        if len(dfs) > 1:
            base_df = dfs[0]
            for df in dfs[1:]:
                # Find common columns for merging (geographic info columns)
                common_cols = [col for col in base_df.columns if col in df.columns and col in excluded_columns]
                if common_cols:
                    base_df = pd.merge(base_df, df, on=common_cols, how='outer', suffixes=('', '_dup'))
                else:
                    # If no common columns, merge on zip_code_tabulation_area if available
                    if 'zip_code_tabulation_area' in base_df.columns and 'zip_code_tabulation_area' in df.columns:
                        base_df = pd.merge(base_df, df, on='zip_code_tabulation_area', how='outer', suffixes=('', '_dup'))
        else:
            base_df = dfs[0]
        
        # Drop duplicate columns
        base_df = base_df.loc[:, ~base_df.columns.duplicated()]
        
        # Reorder columns to put geographic fields at the end
        geo_columns = [col for col in excluded_columns if col in base_df.columns]
        data_columns = [col for col in base_df.columns if col not in geo_columns]
        base_df = base_df[data_columns + geo_columns]
        
        # Store the joined dataframe in session state for download
        st.session_state.joined_df = base_df
        
        st.success("Files joined and merged successfully.")
        st.dataframe(base_df, use_container_width=True)
    
    # Download button for joined file (only show if joined_df exists in session state)
    if 'joined_df' in st.session_state:
        st.markdown(" ", unsafe_allow_html=True)
        
        # Convert to CSV for download
        csv_data = st.session_state.joined_df.to_csv(index=False).encode("utf-8")
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"joined_census_tables_{timestamp}.csv"
        
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            type="primary",
            help="Download the joined census tables as CSV"
        )