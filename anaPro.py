import os
import subprocess
import sys

# Auto-install required packages
required_packages = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'plotly']
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Data Analysis Project", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
for key in ['sidebar_visible', 'df', 'uploaded_file']:
    if key not in st.session_state:
        st.session_state[key] = True if key == 'sidebar_visible' else None

# Toggle button
toggle_label = "" if st.session_state.sidebar_visible else ""
if st.button(f"Menu {toggle_label}"):
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible
    st.rerun()

st.title("Data Analysis Project")

# Sidebar
if st.session_state.sidebar_visible:
    with st.sidebar:
        st.title("Navigation Panel")
        st.session_state.uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], help="Upload a CSV file to begin analysis")
        category = st.radio("Categories", ["Data Overview", "Data Cleaning", "Visualization", "Analysis", "About"], index=0)

# Load data with error handling
if st.session_state.uploaded_file is not None and st.session_state.df is None:
    try:
        st.session_state.uploaded_file.seek(0)
        try:
            st.session_state.df = pd.read_csv(st.session_state.uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
            st.session_state.uploaded_file.seek(0)
            for sep, enc in [(',', 'utf-8'), (';', 'utf-8'), ('\t', 'utf-8'), (',', 'latin-1')]:
                try:
                    st.session_state.df = pd.read_csv(st.session_state.uploaded_file, sep=sep, encoding=enc)
                    if sep != ',': st.warning(f"File read using {'semicolon' if sep == ';' else 'tab' if sep == '\t' else 'latin-1 encoding'}")
                    break
                except:
                    st.session_state.uploaded_file.seek(0)
            
        if st.session_state.df is not None and not st.session_state.df.empty and st.session_state.df.shape[1] > 0:
            st.success(f"Dataset loaded successfully! Shape: {st.session_state.df.shape}")
            st.session_state.uploaded_file.seek(0)
            file_preview = st.session_state.uploaded_file.read(200).decode('utf-8', errors='ignore')
            with st.expander("File Preview (first 200 characters)"):
                st.code(file_preview)
        else:
            st.error("File is empty or has no columns")
            st.session_state.df = None
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Main content
if st.session_state.uploaded_file and st.session_state.df is not None:
    st.write("**Quick Data Preview:**")
    st.dataframe(st.session_state.df.head(3), use_container_width=True)

if st.session_state.df is not None:
    df = st.session_state.df
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Dataset Summary")
        st.metric("Total Rows", df.shape[0])
        st.metric("Total Columns", df.shape[1])
        st.metric("Missing Values", df.isnull().sum().sum())
        st.write("**Data Types:**")
        dtype_counts = df.dtypes.value_counts().reset_index()
        dtype_counts.columns = ['Type', 'Count']
        st.dataframe(dtype_counts, hide_index=True, use_container_width=True)
    
    with col2:
        if category == "Data Overview":
            st.subheader("Data Preview")
            rows_to_display = st.slider("Rows to display", min_value=5, max_value=min(100, len(df)), value=10)
            st.dataframe(df.head(rows_to_display), use_container_width=True)
            
            st.subheader("Column Information")
            selected_col = st.selectbox("Select column", df.columns)
            col_info = pd.DataFrame({
                'Property': ['Data Type', 'Unique Values', 'Missing Values'],
                'Value': [str(df[selected_col].dtype), df[selected_col].nunique(), df[selected_col].isnull().sum()]
            })
            st.dataframe(col_info, hide_index=True, use_container_width=True)
            
        elif category == "Data Cleaning":
            st.subheader("Data Cleaning Tools")
            cleaning_option = st.selectbox("Select cleaning operation", 
                ["Handle Missing Values", "Remove Duplicates", "Change Data Types", "Filter Data", "Rename Columns"])
            
            if cleaning_option == "Handle Missing Values":
                st.write("**Missing Values Summary**")
                missing = df.isnull().sum().reset_index()
                missing.columns = ['Column', 'Missing Count']
                missing_data = missing[missing['Missing Count'] > 0]
                st.dataframe(missing_data) if len(missing_data) > 0 else st.success("No missing values found!")
                
                if len(missing_data) > 0:
                    handle_method = st.radio("Handling Method", ["Drop NA", "Fill with Mean", "Fill with Median", "Fill with Mode", "Custom Value"])
                    
                    def apply_cleaning():
                        df_cleaned = df.copy()
                        if handle_method == "Drop NA":
                            threshold = st.slider("Maximum NA% to drop column", 0, 100, 30)
                            if st.button("Apply"):
                                missing_percent = df.isnull().mean() * 100
                                cols_to_drop = missing_percent[missing_percent > threshold].index
                                df_cleaned = df.drop(columns=cols_to_drop).dropna()
                                st.session_state.df = df_cleaned
                                st.success(f"Dropped {len(cols_to_drop)} columns and NA rows. New shape: {df_cleaned.shape}")
                        
                        elif handle_method in ["Fill with Mean", "Fill with Median", "Fill with Mode"]:
                            numeric_cols_with_na = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()]
                            if len(numeric_cols_with_na) > 0:
                                fill_col = st.selectbox("Select column to fill", numeric_cols_with_na)
                                if st.button("Apply"):
                                    if handle_method == "Fill with Mean":
                                        df_cleaned[fill_col].fillna(df_cleaned[fill_col].mean(), inplace=True)
                                    elif handle_method == "Fill with Median":
                                        df_cleaned[fill_col].fillna(df_cleaned[fill_col].median(), inplace=True)
                                    else:
                                        mode_val = df_cleaned[fill_col].mode()
                                        if len(mode_val) > 0:
                                            df_cleaned[fill_col].fillna(mode_val[0], inplace=True)
                                    st.session_state.df = df_cleaned
                                    st.success(f"Filled missing values in {fill_col}")
                            else:
                                st.warning("No numeric columns with missing values found")
                        
                        elif handle_method == "Custom Value":
                            cols_with_na = df.columns[df.isnull().any()]
                            fill_col = st.selectbox("Select column to fill", cols_with_na)
                            fill_value = st.text_input("Enter fill value")
                            if st.button("Apply") and fill_value:
                                try:
                                    if pd.api.types.is_numeric_dtype(df_cleaned[fill_col]):
                                        fill_value = float(fill_value)
                                    df_cleaned[fill_col].fillna(fill_value, inplace=True)
                                    st.session_state.df = df_cleaned
                                    st.success(f"Filled missing values in {fill_col}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    
                    apply_cleaning()
            
            elif cleaning_option == "Remove Duplicates":
                duplicate_count = df.duplicated().sum()
                st.write(f"Current duplicate rows: {duplicate_count}")
                if duplicate_count > 0 and st.button("Remove Duplicates"):
                    initial_rows = len(df)
                    st.session_state.df = df.drop_duplicates()
                    st.success(f"Removed {initial_rows - len(st.session_state.df)} duplicate rows")
                elif duplicate_count == 0:
                    st.success("No duplicate rows found!")
            
            elif cleaning_option == "Change Data Types":
                col_to_convert = st.selectbox("Select column", df.columns)
                st.write(f"Current type: {df[col_to_convert].dtype}")
                new_type = st.selectbox("Convert to", ["object", "int64", "float64", "datetime64", "category", "bool"])
                
                if st.button("Convert"):
                    try:
                        df_cleaned = df.copy()
                        if new_type == "datetime64":
                            date_format = st.text_input("Enter date format", "%Y-%m-%d")
                            df_cleaned[col_to_convert] = pd.to_datetime(df_cleaned[col_to_convert], format=date_format, errors='coerce')
                        else:
                            df_cleaned[col_to_convert] = df_cleaned[col_to_convert].astype(new_type)
                        st.session_state.df = df_cleaned
                        st.success(f"Converted {col_to_convert} to {new_type}")
                    except Exception as e:
                        st.error(f"Conversion error: {e}")
            
            elif cleaning_option == "Filter Data":
                filter_col = st.selectbox("Select column to filter", df.columns)
                
                if pd.api.types.is_numeric_dtype(df[filter_col]):
                    min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                    range_vals = st.slider("Select range", min_val, max_val, (min_val, max_val))
                    if st.button("Apply Filter"):
                        filtered_df = df[(df[filter_col] >= range_vals[0]) & (df[filter_col] <= range_vals[1])]
                        st.session_state.df = filtered_df
                        st.success(f"Filtered to {len(filtered_df)} rows")
                else:
                    unique_vals = df[filter_col].unique()
                    selected_vals = st.multiselect("Select values to keep", unique_vals, default=list(unique_vals))
                    if st.button("Apply Filter"):
                        filtered_df = df[df[filter_col].isin(selected_vals)]
                        st.session_state.df = filtered_df
                        st.success(f"Filtered to {len(filtered_df)} rows")
            
            elif cleaning_option == "Rename Columns":
                col_to_rename = st.selectbox("Select column to rename", df.columns)
                new_name = st.text_input("New column name", col_to_rename)
                if st.button("Rename") and new_name and new_name != col_to_rename:
                    df_cleaned = df.copy()
                    df_cleaned.rename(columns={col_to_rename: new_name}, inplace=True)
                    st.session_state.df = df_cleaned
                    st.success(f"Renamed {col_to_rename} to {new_name}")
        
        elif category == "Visualization":
            st.subheader("Visualization Tools")
            plot_type = st.selectbox("Select plot type", 
                ["Histogram", "Bar Plot", "Box Plot", "Scatter Plot", "Line Plot", "Heatmap", "Pie Chart"])
            
            def create_plot():
                fig, ax = plt.subplots()
                
                if plot_type in ["Histogram", "Bar Plot", "Box Plot", "Pie Chart"]:
                    col_to_plot = st.selectbox("Select column", df.columns)
                    
                    if plot_type == "Histogram":
                        if pd.api.types.is_numeric_dtype(df[col_to_plot]):
                            bins = st.slider("Number of bins", 5, 100, 20)
                            kde = st.checkbox("Show KDE", True)
                            sns.histplot(df[col_to_plot].dropna(), bins=bins, kde=kde, ax=ax)
                            plt.title(f"Histogram of {col_to_plot}")
                        else:
                            return st.warning("Select a numeric column for histogram")
                    
                    elif plot_type == "Bar Plot":
                        top_n = st.slider("Show top N values", 5, 50, 10)
                        df[col_to_plot].value_counts().head(top_n).plot(kind='bar', ax=ax)
                        plt.title(f"Bar Plot of {col_to_plot}")
                        plt.xticks(rotation=45)
                    
                    elif plot_type == "Box Plot":
                        if pd.api.types.is_numeric_dtype(df[col_to_plot]):
                            sns.boxplot(x=df[col_to_plot].dropna(), ax=ax)
                            plt.title(f"Box Plot of {col_to_plot}")
                        else:
                            return st.warning("Select a numeric column for box plot")
                    
                    elif plot_type == "Pie Chart":
                        top_n = st.slider("Show top N values", 5, 20, 5)
                        df[col_to_plot].value_counts().head(top_n).plot(kind='pie', autopct='%1.1f%%', ax=ax)
                        plt.title(f"Pie Chart of {col_to_plot}")
                        plt.ylabel('')
                
                elif plot_type == "Scatter Plot":
                    num_cols = df.select_dtypes(include=np.number).columns
                    if len(num_cols) >= 2:
                        x_col = st.selectbox("X-axis", num_cols)
                        y_col = st.selectbox("Y-axis", num_cols, index=1)
                        hue_col = st.selectbox("Hue (optional)", [None] + list(df.columns))
                        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
                    else:
                        return st.warning("Need at least 2 numeric columns")
                
                elif plot_type == "Line Plot":
                    num_cols = df.select_dtypes(include=np.number).columns
                    if len(num_cols) >= 1:
                        x_col = st.selectbox("X-axis", df.columns)
                        y_col = st.selectbox("Y-axis", num_cols)
                        sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
                        plt.title(f"Line Plot: {y_col} over {x_col}")
                    else:
                        return st.warning("Need at least 1 numeric column")
                
                elif plot_type == "Heatmap":
                    num_cols = df.select_dtypes(include=np.number).columns
                    if len(num_cols) >= 2:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                        plt.title("Correlation Heatmap")
                    else:
                        return st.warning("Need at least 2 numeric columns")
                
                st.pyplot(fig)
            
            create_plot()
        
        elif category == "Analysis":
            st.subheader("Data Analysis Tools")
            analysis_type = st.selectbox("Select analysis type", 
                ["Descriptive Statistics", "Correlation Analysis", "Group By Analysis", "Pivot Table", "Time Series Analysis"])
            
            if analysis_type == "Descriptive Statistics":
                st.write("**Descriptive Statistics**")
                st.dataframe(df.describe(include='all'))
            
            elif analysis_type == "Correlation Analysis":
                num_cols = df.select_dtypes(include=np.number).columns
                if len(num_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X variable", num_cols)
                    with col2:
                        y_col = st.selectbox("Y variable", num_cols, index=1)
                    
                    clean_data = df[[x_col, y_col]].dropna()
                    if len(clean_data) > 1:
                        corr = clean_data[x_col].corr(clean_data[y_col])
                        st.metric("Pearson Correlation", f"{corr:.3f}")
                        fig, ax = plt.subplots()
                        sns.regplot(data=clean_data, x=x_col, y=y_col, ax=ax)
                        plt.title(f"{x_col} vs {y_col} (r = {corr:.3f})")
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough data points after removing NaN values")
                else:
                    st.warning("Need at least 2 numeric columns")
            
            elif analysis_type == "Group By Analysis":
                group_col = st.selectbox("Group by column", df.columns)
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    agg_col = st.selectbox("Aggregate column", numeric_cols)
                    agg_func = st.selectbox("Aggregation function", ["mean", "sum", "count", "min", "max", "median", "std"])
                    
                    try:
                        grouped = df.groupby(group_col)[agg_col].agg(agg_func).sort_values(ascending=False)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Grouped Results**")
                            st.dataframe(grouped)
                        with col2:
                            st.write("**Visualization**")
                            fig, ax = plt.subplots()
                            grouped.head(10).plot(kind='bar', ax=ax)
                            plt.title(f"{agg_func.title()} of {agg_col} by {group_col}")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("No numeric columns available")
            
            elif analysis_type == "Pivot Table":
                index_col = st.selectbox("Index column", df.columns)
                columns_col = st.selectbox("Columns column (optional)", [None] + list(df.columns))
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    values_col = st.selectbox("Values column", numeric_cols)
                    agg_func = st.selectbox("Aggregation function", ["mean", "sum", "count", "min", "max"])
                    
                    try:
                        pivot = pd.pivot_table(df, index=index_col, columns=columns_col, values=values_col, aggfunc=agg_func, fill_value=0)
                        st.write("**Pivot Table**")
                        st.dataframe(pivot.style.background_gradient(cmap='Blues'))
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("No numeric columns available")
            
            elif analysis_type == "Time Series Analysis":
                datetime_cols = df.select_dtypes(include=['datetime64']).columns
                potential_date_cols = [col for col in df.columns if df[col].dtype == 'object' and 
                                     any(pd.to_datetime(df[col].head(10), errors='coerce').notna())]
                all_date_cols = list(datetime_cols) + potential_date_cols
                
                if len(all_date_cols) >= 1:
                    date_col = st.selectbox("Date column", all_date_cols)
                    numeric_cols = df.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0:
                        value_col = st.selectbox("Value column", numeric_cols)
                        resample_freq = st.selectbox("Resample frequency", ["D", "W", "M", "Q", "Y"], index=2)
                        freq_map = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}
                        
                        try:
                            df_ts = df.copy()
                            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                            df_ts = df_ts.dropna(subset=[date_col, value_col])
                            df_ts.set_index(date_col, inplace=True)
                            resampled = df_ts[value_col].resample(resample_freq).mean()
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            resampled.plot(ax=ax)
                            plt.title(f"{freq_map[resample_freq]} {value_col} over Time")
                            plt.ylabel(value_col)
                            plt.xlabel("Date")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("No numeric columns available")
                else:
                    st.warning("No datetime columns found")
        
        elif category == "About":
            st.subheader("About This Project")
            st.write("**Developer:** Bhagwant Singh")
            st.write("**Tools Used:** Pandas, NumPy, Matplotlib, Seaborn, Streamlit")
            st.subheader("Developer Note")
            st.write("This project was created in Summer training Camp of BFGI, Bathinda.")

else:
    st.info("Please upload a CSV file from the sidebar to begin analysis.")
    st.subheader("While uploading CSV files:")
    st.write("- Ensure your CSV file has proper headers\n- Check that the file is not empty\n- Common separators supported: comma (,), semicolon (;), tab")

# Hide Streamlit style elements
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)