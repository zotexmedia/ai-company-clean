"""Simple Streamlit UI for CSV Company Name Cleaning."""

import io
import pandas as pd
import requests
import streamlit as st
from typing import List, Dict, Any
import time
import concurrent.futures
import threading

# Configure page
st.set_page_config(
    page_title="CSV Company Name Cleaner",
    page_icon="🧹",
    layout="centered"
)

# API Configuration
API_URL = "https://ai-company-clean.onrender.com/normalize"

def process_batch(batch, batch_num, total_batches):
    """Process a single batch of records."""
    try:
        # Extended timeout for GPT-5 Nano processing
        timeout = 180 if len(batch) > 30 else 150
        response = requests.post(
            API_URL,
            json={"records": batch},
            timeout=timeout
        )
        response.raise_for_status()
        return batch_num, response.json()["results"]
    except Exception as e:
        return batch_num, f"Error: {str(e)}"

def clean_company_names(df: pd.DataFrame, company_column: str) -> pd.DataFrame:
    """Clean company names using the API with concurrent processing and duplicate detection."""
    # Detect duplicates and create mapping
    unique_names = {}  # raw_name -> first_occurrence_idx
    duplicate_mapping = {}  # row_idx -> original_row_idx
    
    # Prepare records for API, tracking duplicates
    records = []
    for idx, row in df.iterrows():
        raw_name = str(row[company_column]).strip()
        
        if raw_name in unique_names:
            # This is a duplicate - map it to the original
            duplicate_mapping[idx] = unique_names[raw_name]
        else:
            # First occurrence - add to processing queue
            unique_names[raw_name] = idx
            records.append({
                "id": f"row-{idx}",
                "raw_name": raw_name,
                "source": "csv",
                "country_hint": "US"
            })
    
    st.info(f"🔄 Found {len(df) - len(records)} duplicates. Processing {len(records)} unique names instead of {len(df)}.")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process in concurrent batches - balanced for speed and reliability
    batch_size = 50   # Larger batches for efficiency  
    max_workers = 6   # More conservative concurrency to prevent timeouts
    
    # Create batches
    batches = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batches.append(batch)
    
    all_results = [None] * len(batches)
    completed = 0
    
    # Process batches concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_batch, batch, i, len(batches)): i 
            for i, batch in enumerate(batches)
        }
        
        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            batch_num, result = future.result()
            
            if isinstance(result, str):  # Error case
                st.error(f"Error processing batch {batch_num + 1}: {result}")
                return None
                
            all_results[batch_num] = result
            completed += 1
            
            # Update progress
            progress = completed / len(batches)
            progress_bar.progress(progress)
            status_text.text(f"Completed {completed} of {len(batches)} batches ({completed * batch_size} companies)...")
    
    # Flatten results
    flattened_results = []
    for batch_results in all_results:
        flattened_results.extend(batch_results)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Create results mapping from API responses
    results_by_idx = {}
    for result in flattened_results:
        row_idx = int(result["id"].split("-")[1])
        results_by_idx[row_idx] = result
    
    # Create results DataFrame, handling duplicates
    cleaned_data = []
    for idx, row in df.iterrows():
        original_row = row.copy()
        
        # Check if this row is a duplicate
        if idx in duplicate_mapping:
            # Use result from the original occurrence
            source_idx = duplicate_mapping[idx]
            result = results_by_idx.get(source_idx)
        else:
            # Use direct result
            result = results_by_idx.get(idx)
        
        if result and result.get("result"):
            original_row["Cleaned_Company_Name"] = result["result"]["canonical"]
        else:
            original_row["Cleaned_Company_Name"] = "ERROR"
        
        cleaned_data.append(original_row)
    
    return pd.DataFrame(cleaned_data)

def main():
    # Header
    st.markdown("# 🧹 CSV Company Name Cleaner")
    st.markdown("Upload a CSV, pick the column with company names, and download a cleaned file.")
    
    # Features list
    st.markdown("""
    • **Hyphens and ampersands preserved** (e.g., Jan-Pro, H&H, Eldredge & Clark)
    • **Legal/prof suffixes removed** (Inc, LLC, LLP, PC, DDS, ...)
    • **Stop-words lowercase** inside the name but capitalized if first (The Barcode Group)
    • **State codes and whitelisted acronyms** stay uppercase (LTB, USA...)
    • **Optional fuzzy de-duplication** to group near-identical names into a *Canonical Company Name* column
    """)
    
    st.markdown("---")
    
    # File upload section
    st.markdown("### Upload CSV")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=['csv'],
        help="Limit 200MB per file • CSV"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Column selection
            st.markdown("### Select Company Name Column")
            company_column = st.selectbox(
                "Choose the column containing company names:",
                df.columns.tolist()
            )
            
            # Preview
            if company_column:
                st.markdown("### Preview")
                st.write("First 5 rows:")
                st.dataframe(df[[company_column]].head(), use_container_width=True)
                
                # Process button
                if st.button("🧹 Clean Company Names", type="primary"):
                    # Start timer
                    start_time = time.time()
                    
                    with st.spinner("Processing..."):
                        result_df = clean_company_names(df, company_column)
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    if result_df is not None:
                        st.success(f"✅ Processing complete! Took {processing_time:.1f} seconds")
                        
                        # Show results preview
                        st.markdown("### Results Preview")
                        preview_cols = [company_column, "Cleaned_Company_Name"]
                        available_cols = [col for col in preview_cols if col in result_df.columns]
                        st.dataframe(result_df[available_cols].head(10), use_container_width=True)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Cleaned CSV",
                            data=csv_data,
                            file_name=f"cleaned_{uploaded_file.name}",
                            mime="text/csv",
                            type="primary"
                        )
                        
                        # Stats
                        total_processed = len(result_df)
                        successful_cleanings = len(result_df[result_df["Cleaned_Company_Name"] != "ERROR"])
                        processing_rate = total_processed / processing_time if processing_time > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Processed", total_processed)
                        with col2:
                            st.metric("Successful Cleanings", successful_cleanings)
                        with col3:
                            st.metric("Processing Rate", f"{processing_rate:.1f} companies/sec")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.error("Please make sure your file is a valid CSV format.")

if __name__ == "__main__":
    main()