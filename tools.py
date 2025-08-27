import pandas as pd
import numpy as np
from IPython.display import Markdown, display


def print_markdown(data):
    """
    Print data as markdown. Handles strings, dictionaries, and lists.

    Parameters:
    - data: str, dict, or list - The data to display as markdown
    """
    if isinstance(data, str):
        # Original string functionality
        display(Markdown(data))
    elif isinstance(data, dict):
        # Convert dictionary to markdown table
        markdown_table = _dict_to_markdown_table(data)
        display(Markdown(markdown_table))
    elif isinstance(data, list):
        # Convert list to markdown bullet points
        markdown_list = _list_to_markdown_bullets(data)
        display(Markdown(markdown_list))
    else:
        # For other types, convert to string first
        display(Markdown(str(data)))


def _dict_to_markdown_table(data):
    """Convert dictionary to markdown table format"""
    if not data:
        return "| Key | Value |\n|-----|-------|\n| (empty) | (empty) |"

    # Handle nested dictionaries and complex values
    table_rows = []
    table_rows.append("| Key | Value |")
    table_rows.append("|-----|-------|")

    for key, value in data.items():
        # Format the value based on its type
        if isinstance(value, (dict, list)):
            # For nested structures, convert to string representation
            formatted_value = str(value).replace('|', '\\|')
        elif value is None:
            formatted_value = "*None*"
        elif isinstance(value, str):
            # Escape pipe characters in strings
            formatted_value = value.replace('|', '\\|')
        else:
            formatted_value = str(value).replace('|', '\\|')

        # Escape pipe characters in keys too
        formatted_key = str(key).replace('|', '\\|')
        table_rows.append(f"| {formatted_key} | {formatted_value} |")

    return "\n".join(table_rows)


def _list_to_markdown_bullets(data):
    """Convert list to markdown bullet points"""
    if not data:
        return "• *(empty list)*"

    bullet_points = []
    for item in data:
        if isinstance(item, (dict, list)):
            # For nested structures, indent them
            formatted_item = str(item)
        elif item is None:
            formatted_item = "*None*"
        else:
            formatted_item = str(item)

        bullet_points.append(f"• {formatted_item}")

    return "\n".join(bullet_points)


def load_dataset(file_path, separator=";"):
    print_markdown("\n" + "=" * 70)
    print_markdown("📋 STEP 1: LOAD  DATASET FROM CSV")
    print_markdown("=" * 70)
    return  pd.read_csv(file_path, sep=separator)

def get_basic_info(df):
    """
    STEP 2: Get comprehensive basic information about the dataset

    WHY THIS IS IMPORTANT:
    - Understanding data types helps us choose appropriate analysis methods
    - Memory usage information helps with performance optimization
    - Knowing the scale of data helps plan computational resources

    WHAT IT ANALYZES:
    - Data types and memory usage
    - Dataset size and structure
    - Quick overview of numerical vs categorical data

    Parameters:
    - df (DataFrame): Input dataset

    Returns:
    - dict: Summary information about the dataset
    """
    print_markdown("\n" + "=" * 70)
    print_markdown("📋 STEP 2: BASIC DATASET INFORMATION")
    print_markdown("=" * 70)

    # Dataset dimensions
    n_rows, n_cols = df.shape
    print_markdown(f"📐 Dataset Dimensions:")
    print_markdown(f"   • Rows (samples): {n_rows:,}")
    print_markdown(f"   • Columns (features): {n_cols}")

    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2  # Convert to MB
    print_markdown(f"   • Memory usage: {memory_usage:.2f} MB")

    # Data type distribution
    dtype_counts = df.dtypes.value_counts()
    print_markdown(f"\n🏷️  Data Type Distribution:")
    for dtype, count in dtype_counts.items():
        print_markdown(f"   • {dtype}: {count} columns")

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print_markdown("\n📊 Column Classification:")
    num_cols_str = ", ".join(numerical_cols) if numerical_cols else "(none)"
    cat_cols_str = ", ".join(categorical_cols) if categorical_cols else "(none)"
    table = (
        "| Type | Count | Columns |\n"
        "|:------|:-------|:---------|\n"
        f"| **Numerical** | **{len(numerical_cols)}** | {num_cols_str} |\n"
        f"| **Categorical** | **{len(categorical_cols)}** | {cat_cols_str} |"
    )
    print_markdown(table)

    # Return summary for further use
    summary = {
        'shape': df.shape,
        'memory_mb': memory_usage,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'dtypes': dict(df.dtypes)
    }

    return summary


def check_data_quality(df):
    """
    STEP 3: Comprehensive data quality assessment

    WHY DATA QUALITY MATTERS:
    - Missing values can bias analysis and models
    - Duplicates can inflate importance of certain patterns
    - Data inconsistencies can lead to wrong conclusions
    - Understanding quality issues helps decide on preprocessing steps

    WHAT IT CHECKS:
    - Missing values (null, NaN, empty strings)
    - Duplicate rows
    - Data consistency issues
    - Potential data entry errors

    Parameters:
    - df (DataFrame): Input dataset

    Returns:
    - dict: Data quality report
    """
    print_markdown("\n" + "=" * 70)
    print_markdown("🔍 STEP 3: DATA QUALITY ASSESSMENT")
    print_markdown("=" * 70)

    quality_report = {}

    # 1. Missing Values Analysis
    print_markdown("1️⃣ Missing Values Analysis:")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    has_missing = missing_counts.sum() > 0

    if has_missing:
        print_markdown("   ⚠️  Missing values detected:")
        for col in missing_counts.index:
            if missing_counts[col] > 0:
                print_markdown(f"      • {col}: {missing_counts[col]} ({missing_percentages[col]:.1f}%)")
    else:
        print_markdown("   ✅ No missing values found!")

    quality_report['missing_values'] = missing_counts.to_dict()

    # 2. Duplicate Rows
    print_markdown(f"\n2️⃣ Duplicate Rows Analysis:")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print_markdown(f"   ⚠️  Found {duplicate_count} duplicate rows ({duplicate_count / len(df) * 100:.1f}%)")
    else:
        print_markdown("   ✅ No duplicate rows found!")

    quality_report['duplicates'] = duplicate_count

    # 3. Check for empty strings in categorical columns
    print_markdown(f"\n3️⃣ Empty String Analysis:")
    empty_strings = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        empty_count = (df[col] == '').sum()
        if empty_count > 0:
            empty_strings[col] = empty_count
            print_markdown(f"   ⚠️  {col}: {empty_count} empty strings")

    if not empty_strings:
        print_markdown("   ✅ No empty strings found in categorical columns!")

    quality_report['empty_strings'] = empty_strings

    # 4. Data consistency checks
    print_markdown(f"\n4️⃣ Data Consistency Checks:")
    consistency_issues = []

    # Check for mixed case in categorical variables
    for col in categorical_cols:
        unique_values = df[col].dropna().unique()
        if len(unique_values) != len([str(val).lower() for val in set(str(val).lower() for val in unique_values)]):
            consistency_issues.append(f"{col}: Mixed case values detected")

    # Check for unusual numerical values (negative where shouldn't be, etc.)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col.lower() in ['age', 'duration', 'campaign'] and (df[col] < 0).any():
            consistency_issues.append(f"{col}: Contains negative values")

    if consistency_issues:
        for issue in consistency_issues:
            print_markdown(f"   ⚠️  {issue}")
    else:
        print_markdown("   ✅ No obvious consistency issues detected!")

    quality_report['consistency_issues'] = consistency_issues

    # 5. Summary
    print_markdown(f"\n📊 Data Quality Summary:")
    print_markdown(f"   • Dataset completeness: {(1 - missing_counts.sum() / (len(df) * len(df.columns))) * 100:.1f}%")
    print_markdown(f"   • Unique rows: {len(df) - duplicate_count}/{len(df)}")

    return quality_report