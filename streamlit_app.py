"""Simple Streamlit UI for CSV Company Name Cleaning."""

import io
import pandas as pd
import requests
import streamlit as st
from typing import List, Dict, Any
import time

# Configure page
st.set_page_config(
    page_title="CSV Company Name Cleaner",
    page_icon="🧹",
    layout="centered"
)

# API Configuration
API_URL = "https://ai-company-clean.onrender.com/normalize"

def clean_company_names(df: pd.DataFrame, company_column: str) -> pd.DataFrame:
    """Clean company names using the API."""
    # Prepare records for API
    records = []
    for idx, row in df.iterrows():
        records.append({
            "id": f"row-{idx}",
            "raw_name": str(row[company_column]),
            "source": "csv",
            "country_hint": "US"
        })
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process in batches
    batch_size = 50
    all_results = []
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        status_text.text(f"Processing {i + 1}-{min(i + batch_size, len(records))} of {len(records)} companies...")
        
        try:
            response = requests.post(
                API_URL,
                json={"records": batch},
                timeout=120
            )
            response.raise_for_status()
            
            batch_results = response.json()["results"]
            all_results.extend(batch_results)
            
            # Update progress
            progress = min((i + batch_size) / len(records), 1.0)
            progress_bar.progress(progress)
            
            # Small delay to be nice to the API
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            return None
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Create results DataFrame
    cleaned_data = []
    for result in all_results:
        row_idx = int(result["id"].split("-")[1])
        original_row = df.iloc[row_idx].copy()
        
        if result.get("result"):
            original_row["Cleaned_Company_Name"] = result["result"]["canonical"]
            original_row["Confidence"] = result["result"]["confidence"]
            original_row["Reason"] = result["result"]["reason"]
        else:
            original_row["Cleaned_Company_Name"] = "ERROR"
            original_row["Confidence"] = 0.0
            original_row["Reason"] = result.get("error", "Unknown error")
        
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
                    with st.spinner("Processing..."):
                        result_df = clean_company_names(df, company_column)
                    
                    if result_df is not None:
                        st.success("✅ Processing complete!")
                        
                        # Show results preview
                        st.markdown("### Results Preview")
                        preview_cols = [company_column, "Cleaned_Company_Name", "Confidence", "Reason"]
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
                        high_confidence = len(result_df[result_df["Confidence"] >= 0.9])
                        total_processed = len(result_df)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Processed", total_processed)
                        with col2:
                            st.metric("High Confidence (≥90%)", high_confidence)
                        with col3:
                            st.metric("Success Rate", f"{(high_confidence/total_processed)*100:.1f}%")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.error("Please make sure your file is a valid CSV format.")

if __name__ == "__main__":
    main()