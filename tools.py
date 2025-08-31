import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency


def add_percentage_labels(ax, series):
    """
    Añade etiquetas de porcentaje sobre las barras de un gráfico de seaborn.
    (Adds percentage labels on top of the bars of a seaborn plot.)

    Args:
        ax (matplotlib.axes.Axes): El objeto de ejes del gráfico. (The axes object of the plot.)
        series (pd.Series): La serie de pandas con los datos para calcular el total. (The pandas series with the data to calculate the total.)
    """
    total = len(series)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=12, color='black')


def plot_distribution(dataframe, column_name):
    """
    Crea y muestra un gráfico de barras de la distribución de una columna categórica.
    (Creates and displays a bar plot of the distribution of a categorical column.)

    Args:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos. (The DataFrame containing the data.)
        column_name (str): El nombre de la columna a visualizar. (The name of the column to visualize.)
    """
    # Configuramos el estilo y tamaño del gráfico.
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # Calculamos el orden de las barras (de mayor a menor).
    order = dataframe[column_name].value_counts().index

    # Creamos el gráfico de barras.
    ax = sns.countplot(x=column_name, data=dataframe, order=order, palette='viridis')

    # Añadimos los porcentajes sobre cada barra.
    add_percentage_labels(ax, dataframe[column_name])

    # Añadimos títulos y etiquetas.
    plt.title(
        f'Distribución de Clientes por {column_name.capitalize()} (Client Distribution by {column_name.capitalize()})',
        fontsize=16)
    plt.xlabel(f'{column_name.capitalize()}', fontsize=14)
    plt.ylabel('Cantidad de Clientes (Number of Clients)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Mostramos el gráfico.
    plt.show()


def print_section_header(title, char='-'):
    """Print formatted section header."""
    print(f"\n{title}")
    print(char * len(title))


# =============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath):
    """
    Load dataset and perform basic preprocessing.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    df = pd.read_csv(filepath, sep=';')
    df.columns = df.columns.str.replace('"', '')
    return df


def get_basic_info(df):
    """
    Extract basic dataset information.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Dictionary with basic dataset info
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum().to_dict()
    }


# =============================================================================
# STEP 2: DATA QUALITY ANALYSIS
# =============================================================================

def analyze_data_quality(df):
    """
    Analyze data quality metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Data quality summary
    """
    quality_summary = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null_count': df.isnull().sum(),
        'null_percentage': (df.isnull().sum() / len(df)) * 100,
        'unique_values': df.nunique(),
        'memory_usage': df.memory_usage(deep=True)
    })

    return quality_summary.round(2)


def identify_variable_types(df):
    """
    Categorize variables by type and variability.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Dictionary with variable categorization
    """
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_vars = df.select_dtypes(include=['object']).columns.tolist()

    # Variables with low variability
    low_variability = []
    for col in df.columns:
        if df[col].nunique() <= 3:
            low_variability.append({
                'variable': col,
                'unique_count': df[col].nunique(),
                'unique_values': df[col].unique().tolist()
            })

    return {
        'numeric': numeric_vars,
        'categorical': categorical_vars,
        'low_variability': low_variability
    }


# =============================================================================
# STEP 3: TARGET VARIABLE ANALYSIS
# =============================================================================

def analyze_target_variable(df, target_col='y'):
    """
    Comprehensive analysis of target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column

    Returns:
    --------
    dict
        Target variable analysis results
    """
    target_counts = df[target_col].value_counts()
    target_props = df[target_col].value_counts(normalize=True) * 100

    return {
        'counts': target_counts.to_dict(),
        'proportions': target_props.to_dict(),
        'is_balanced': min(target_props) >= 30,  # Threshold for balance
        'minority_class': target_props.idxmin(),
        'majority_class': target_props.idxmax(),
        'imbalance_ratio': target_props.max() / target_props.min()
    }


# =============================================================================
# STEP 4: NUMERIC VARIABLES ANALYSIS
# =============================================================================

def analyze_numeric_variables(df, numeric_vars):
    """
    Analyze numeric variables with descriptive statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_vars : list
        List of numeric variable names

    Returns:
    --------
    dict
        Numeric variables analysis
    """
    if not numeric_vars:
        return {'stats': None, 'variable_vars': []}

    # Basic statistics
    numeric_stats = df[numeric_vars].describe()

    # Variables with actual variability
    variable_vars = [col for col in numeric_vars if df[col].nunique() > 1]

    # Distribution analysis
    distribution_info = {}
    for col in variable_vars:
        data = df[col].dropna()
        distribution_info[col] = {
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'normality_test': stats.normaltest(data)[1] if len(data) > 8 else None
        }

    return {
        'stats': numeric_stats,
        'variable_vars': variable_vars,
        'distribution_info': distribution_info
    }


# =============================================================================
# STEP 5: CATEGORICAL VARIABLES ANALYSIS
# =============================================================================

def analyze_categorical_variables(df, categorical_vars, target_col='y'):
    """
    Analyze categorical variables and their relationship with target.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_vars : list
        List of categorical variable names
    target_col : str
        Name of target column

    Returns:
    --------
    dict
        Categorical variables analysis
    """
    analysis = {}

    for col in categorical_vars:
        if col != target_col:
            # Basic frequency analysis
            value_counts = df[col].value_counts()
            proportions = df[col].value_counts(normalize=True) * 100

            # Conversion rates by category
            conversion_rates = df.groupby(col)[target_col].apply(
                lambda x: (x == 'yes').sum() / len(x) * 100
            ).round(2)

            analysis[col] = {
                'value_counts': value_counts.to_dict(),
                'proportions': proportions.to_dict(),
                'conversion_rates': conversion_rates.to_dict(),
                'unique_count': df[col].nunique()
            }

    return analysis


# =============================================================================
# STEP 6: BIVARIATE ANALYSIS
# =============================================================================

def analyze_numeric_vs_target(df, numeric_var, target_col='y'):
    """
    Analyze relationship between numeric variable and target.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_var : str
        Name of numeric variable
    target_col : str
        Name of target column

    Returns:
    --------
    dict
        Analysis results
    """
    # Group statistics
    group_stats = df.groupby(target_col)[numeric_var].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    # Statistical test
    groups = [df[df[target_col] == val][numeric_var].values
              for val in df[target_col].unique()]

    if len(groups) == 2 and all(len(g) > 0 for g in groups):
        t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
        test_result = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    else:
        test_result = None

    # Percentile analysis
    percentiles = {}
    for target_val in df[target_col].unique():
        data = df[df[target_col] == target_val][numeric_var]
        percentiles[target_val] = {
            f'p{p}': np.percentile(data, p) for p in [25, 50, 75, 90, 95]
        }

    return {
        'group_stats': group_stats,
        'statistical_test': test_result,
        'percentiles': percentiles
    }


def analyze_categorical_vs_target(df, categorical_var, target_col='y'):
    """
    Analyze relationship between categorical variable and target.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_var : str
        Name of categorical variable
    target_col : str
        Name of target column

    Returns:
    --------
    dict
        Analysis results
    """
    # Contingency table
    crosstab = pd.crosstab(df[categorical_var], df[target_col])

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(crosstab)

    # Conversion rates
    conversion_rates = df.groupby(categorical_var)[target_col].apply(
        lambda x: (x == 'yes').sum() / len(x) * 100
    ).round(2).sort_values(ascending=False)

    return {
        'crosstab': crosstab,
        'chi2_test': {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'significant': p_value < 0.05
        },
        'conversion_rates': conversion_rates
    }


# =============================================================================
# STEP 7: CORRELATION ANALYSIS
# =============================================================================

def calculate_correlations(df, numeric_vars):
    """
    Calculate correlation matrix for numeric variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_vars : list
        List of numeric variables

    Returns:
    --------
    dict
        Correlation analysis results
    """
    # Only use variables with variability
    variable_vars = [col for col in numeric_vars if df[col].nunique() > 1]

    if len(variable_vars) < 2:
        return {'correlation_matrix': None, 'strong_correlations': []}

    corr_matrix = df[variable_vars].corr().round(3)

    # Find strong correlations (|r| > 0.5)
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })

    return {
        'correlation_matrix': corr_matrix,
        'strong_correlations': strong_correlations,
        'variable_vars': variable_vars
    }


# =============================================================================
# STEP 8: VISUALIZATION FUNCTIONS
# =============================================================================

# =============================================================================
# STEP 8: VISUALIZATION FUNCTIONS - ALL REQUIRED TYPES
# =============================================================================
# =============================================================================
# STEP 8: VISUALIZATION FUNCTIONS - ALL REQUIRED TYPES
# =============================================================================

def create_scatter_plots_numeric(df, numeric_var, target_col='y'):
    """1. DIAGRAMAS DE DISPERSIÓN (para variables numéricas vs target)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DIAGRAMAS DE DISPERSIÓN (Variable Numérica vs Target)', fontsize=16, fontweight='bold', y=1.02)

    # Create a numeric encoding of target for scatter plot
    target_numeric = df[target_col].map({'no': 0, 'yes': 1})

    # 1. Basic scatter plot: numeric_var vs target (with jitter for binary target)
    jitter = np.random.normal(0, 0.05, len(df))
    for target_val, color in zip(['no', 'yes'], ['lightcoral', 'lightblue']):
        mask = df[target_col] == target_val
        y_pos = (target_numeric[mask] + jitter[mask]).values
        axes[0, 0].scatter(df.loc[mask, numeric_var], y_pos,
                           alpha=0.6, s=40, label=f'y={target_val}', c=color)

    axes[0, 0].set_xlabel(numeric_var)
    axes[0, 0].set_ylabel(f'{target_col} (con jitter)')
    axes[0, 0].set_title(f'{numeric_var} vs {target_col}')
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_yticklabels(['no', 'yes'])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot style scatter (strip plot)
    for i, target_val in enumerate(['no', 'yes']):
        mask = df[target_col] == target_val
        y_pos = np.full(mask.sum(), i) + np.random.normal(0, 0.05, mask.sum())
        color = 'lightcoral' if target_val == 'no' else 'lightblue'
        axes[0, 1].scatter(df.loc[mask, numeric_var], y_pos,
                           alpha=0.6, s=30, c=color)

    axes[0, 1].set_xlabel(numeric_var)
    axes[0, 1].set_ylabel(target_col)
    axes[0, 1].set_title(f'Distribución de {numeric_var} por {target_col}')
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_yticklabels(['no', 'yes'])
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Histogram overlay
    for target_val, color in zip(['no', 'yes'], ['lightcoral', 'lightblue']):
        mask = df[target_col] == target_val
        axes[1, 0].hist(df.loc[mask, numeric_var], alpha=0.6,
                        label=f'y={target_val}', color=color, bins=20)

    axes[1, 0].set_xlabel(numeric_var)
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title(f'Histograma de {numeric_var} por {target_col}')
    axes[1, 0].legend()

    # 4. Density plot / KDE approximation with binned data
    for target_val, color in zip(['no', 'yes'], ['lightcoral', 'lightblue']):
        mask = df[target_col] == target_val
        data = df.loc[mask, numeric_var]

        # Create bins and compute density
        bins = np.linspace(data.min(), data.max(), 15)
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        axes[1, 1].plot(bin_centers, hist, color=color, alpha=0.8,
                        linewidth=2, label=f'y={target_val}')
        axes[1, 1].fill_between(bin_centers, hist, alpha=0.3, color=color)

    axes[1, 1].set_xlabel(numeric_var)
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].set_title(f'Densidad de {numeric_var} por {target_col}')
    axes[1, 1].legend()

    # Add correlation info
    correlation = np.corrcoef(df[numeric_var], target_numeric)[0, 1]
    fig.text(0.02, 0.02, f'Correlación {numeric_var} - {target_col}: {correlation:.3f}',
             fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    return fig


def create_target_distribution_plot(target_analysis):
    """Create target variable distribution plot for context."""
    fig, ax = plt.subplots(1, 1, figsize=(16,12))


    counts = target_analysis['counts']
    proportions = target_analysis['proportions']

    bars = ax.bar(counts.keys(), counts.values(),
                  color=['lightcoral', 'lightblue'], alpha=0.8, edgecolor='black')
    ax.set_title('Distribución de Variable Objetivo', fontsize=28, fontweight='bold')
    ax.set_xlabel('Target Value (y)')
    ax.set_ylabel('Frecuencia')

    # Add percentage and count labels
    for bar, (key, prop) in zip(bars, proportions.items()):
        height = bar.get_height()
        count = counts[key]
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(counts.values()) * 0.01,
                f'{count}\n({prop:.1f}%)', ha='center', va='bottom', fontweight='bold')

    return fig



def create_comparative_boxplots(df, numeric_var, categorical_var, target_col='y'):
    """2. BOXPLOTS COMPARATIVOS (para una variable numérica y una categórica)."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    fig.suptitle('BOXPLOTS COMPARATIVOS (Numérica vs Categórica)', fontsize=16, fontweight='bold', y=1.02)


    # Boxplot: numeric_var by target
    df.boxplot(column=numeric_var, by=target_col, ax=axes[0, 0])
    axes[0, 0].set_title(f'{numeric_var} by {target_col}')


    axes[0, 0].set_ylabel(numeric_var)

    # Boxplot: numeric_var by categorical_var
    if len(df[categorical_var].unique()) <= 10:  # Only if manageable number of categories
        df.boxplot(column=numeric_var, by=categorical_var, ax=axes[0, 1])
        axes[0, 1].set_title(f'{numeric_var} by {categorical_var}')

        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'Too many categories\nfor boxplot display',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title(f'{categorical_var} has too many categories')

    # Violin plot comparison
    if 'y' in df.columns:
        for i, target_val in enumerate(['no', 'yes']):
            data = df[df[target_col] == target_val][numeric_var].dropna()
            if len(data) > 0:
                axes[1, 0].violinplot([data], positions=[i], showmeans=True, showmedians=True)
        axes[1, 0].set_title(f'{numeric_var} Distribution by {target_col}')
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_xticklabels(['no', 'yes'])
        axes[1, 0].set_ylabel(numeric_var)

    # Statistical summary
    stats_by_target = df.groupby(target_col)[numeric_var].agg(['count', 'mean', 'median']).round(2)
    stats_by_target.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title(f'{numeric_var} Statistics by {target_col}')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig


def create_correlation_heatmap(correlation_analysis):
    """3. HEATMAPS DE CORRELACIÓN (para múltiples variables numéricas)."""
    corr_matrix = correlation_analysis['correlation_matrix']

    if corr_matrix is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Insufficient numeric variables\nwith variability for correlation heatmap',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('HEATMAP DE CORRELACIÓN (Variables Numéricas)', fontweight='bold')
        return fig

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                ax=ax,
                cbar_kws={'shrink': 0.8},
                fmt='.3f')

    ax.set_title('HEATMAP DE CORRELACIÓN (Variables Numéricas)', fontsize=14, fontweight='bold', pad=20)

    # Add interpretation guide
    textstr = 'Interpretación:\n|r| < 0.3: Débil\n0.3 ≤ |r| < 0.7: Moderada\n|r| ≥ 0.7: Fuerte'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig


def create_categorical_bar_charts(df, categorical_var, target_col='y'):
    """4. GRÁFICOS DE BARRAS o STACKED CHARTS (para variables categóricas)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GRÁFICOS DE BARRAS Y STACKED CHARTS (Variables Categóricas)',
                 fontsize=30, fontweight='bold', y=1.02)

    # 1. Simple bar chart - frequency distribution
    value_counts = df[categorical_var].value_counts()
    axes[0, 0].bar(range(len(value_counts)), value_counts.values,
                   color='lightsteelblue', alpha=0.8, edgecolor='black')
    axes[0, 0].set_title(f'Distribución de Frecuencias: {categorical_var}')
    axes[0, 0].set_xlabel(categorical_var)
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_xticks(range(len(value_counts)))
    axes[0, 0].set_xticklabels(value_counts.index, rotation=45, ha='right')

    # Add count labels
    for i, v in enumerate(value_counts.values):
        axes[0, 0].text(i, v + max(value_counts.values) * 0.01, str(v),
                        ha='center', va='bottom')

    # 2. Stacked bar chart - categorical_var by target
    crosstab = pd.crosstab(df[categorical_var], df[target_col])
    crosstab.plot(kind='bar', stacked=True, ax=axes[0, 1],
                  color=['lightcoral', 'lightblue'], alpha=0.8)
    axes[0, 1].set_title(f'Stacked Chart: {categorical_var} by {target_col}')
    axes[0, 1].set_xlabel(categorical_var)
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend(title=target_col)

    # 3. Horizontal bar chart - conversion rates
    conversion_rates = df.groupby(categorical_var)[target_col].apply(
        lambda x: (x == 'yes').sum() / len(x) * 100
    ).sort_values(ascending=True)

    axes[1, 0].barh(range(len(conversion_rates)), conversion_rates.values,
                    color='lightgreen', alpha=0.8)
    axes[1, 0].set_title(f'Tasas de Conversión por {categorical_var}')
    axes[1, 0].set_xlabel('Tasa de Conversión (%)')
    axes[1, 0].set_ylabel(categorical_var)
    axes[1, 0].set_yticks(range(len(conversion_rates)))
    axes[1, 0].set_yticklabels(conversion_rates.index)

    # Add percentage labels
    for i, v in enumerate(conversion_rates.values):
        axes[1, 0].text(v + max(conversion_rates.values) * 0.01, i, f'{v:.1f}%',
                        va='center', ha='left')

    # 4. Normalized stacked chart (100% stacked)
    crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    crosstab_norm.plot(kind='bar', stacked=True, ax=axes[1, 1],
                       color=['lightcoral', 'lightblue'], alpha=0.8)
    axes[1, 1].set_title(f'Stacked Chart Normalizado: {categorical_var} by {target_col}')
    axes[1, 1].set_xlabel(categorical_var)
    axes[1, 1].set_ylabel('Porcentaje (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title=target_col)

    plt.tight_layout()
    return fig


# =============================================================================
# STEP 9: REPORTING FUNCTIONS
# =============================================================================

def generate_data_quality_report(quality_analysis, var_types):
    """Generate data quality report."""
    print_section_header("DATA QUALITY REPORT", "=")

    print(f"Dataset Shape: {quality_analysis.shape}")
    print(f"Total Memory Usage: {quality_analysis['memory_usage'].sum() / 1024:.1f} KB")
    print()

    print("Variable Types:")
    print(f"  Numeric: {len(var_types['numeric'])} variables")
    print(f"  Categorical: {len(var_types['categorical'])} variables")
    print()

    if var_types['low_variability']:
        print("Low Variability Variables:")
        for var_info in var_types['low_variability']:
            print(f"  {var_info['variable']}: {var_info['unique_count']} unique values")

    print("\nData Quality Summary:")
    print(quality_analysis)


def generate_target_analysis_report(target_analysis):
    """Generate target variable analysis report."""
    print_section_header("TARGET VARIABLE ANALYSIS", "=")

    print("Distribution:")
    for value, count in target_analysis['counts'].items():
        prop = target_analysis['proportions'][value]
        print(f"  {value}: {count} ({prop:.2f}%)")

    print(f"\nDataset Balance: {'Balanced' if target_analysis['is_balanced'] else 'Imbalanced'}")
    print(f"Imbalance Ratio: {target_analysis['imbalance_ratio']:.1f}:1")
    print(f"Minority Class: {target_analysis['minority_class']}")


def generate_bivariate_report(numeric_analysis, categorical_analysis,
                              selected_numeric, selected_categorical):
    """Generate bivariate analysis report."""
    print_section_header("BIVARIATE ANALYSIS REPORT", "=")

    # Numeric variable analysis
    if numeric_analysis:
        print(f"\n{selected_numeric.upper()} vs TARGET:")
        print("Group Statistics:")
        print(numeric_analysis['group_stats'])

        if numeric_analysis['statistical_test']:
            test = numeric_analysis['statistical_test']
            print(f"\nStatistical Test (t-test):")
            print(f"  t-statistic: {test['t_statistic']:.4f}")
            print(f"  p-value: {test['p_value']:.4f}")
            print(f"  Significant: {'Yes' if test['significant'] else 'No'}")

    # Categorical variable analysis
    if categorical_analysis:
        print(f"\n\n{selected_categorical.upper()} vs TARGET:")
        print("Conversion Rates:")
        for category, rate in categorical_analysis['conversion_rates'].items():
            print(f"  {category}: {rate:.2f}%")

        test = categorical_analysis['chi2_test']
        print(f"\nChi-square Test:")
        print(f"  Chi-square: {test['chi2']:.4f}")
        print(f"  p-value: {test['p_value']:.4f}")
        print(f"  Significant: {'Yes' if test['significant'] else 'No'}")


def generate_summary_report(basic_info, target_analysis, correlation_analysis):
    """Generate final summary report."""
    print_section_header("EXECUTIVE SUMMARY", "=")

    shape = basic_info['shape']
    print(f"Dataset: {shape[0]} records x {shape[1]} variables")
    print(f"Target Distribution: {target_analysis['proportions']}")
    print(f"Data Quality: {'Good' if sum(basic_info['missing_values'].values()) == 0 else 'Issues detected'}")

    if correlation_analysis['strong_correlations']:
        print(f"\nStrong Correlations Found: {len(correlation_analysis['strong_correlations'])}")
        for corr in correlation_analysis['strong_correlations']:
            print(f"  {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}")
    else:
        print("\nNo strong correlations detected")

    print(f"\nLimitations:")
    if target_analysis['imbalance_ratio'] > 10:
        print("  - Severe class imbalance detected")
    if shape[0] < 1000:
        print("  - Small sample size may limit statistical power")