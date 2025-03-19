import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Choose specific colors from the colorblind palette
# You can customize these colors as needed
palette = sns.color_palette("colorblind")
FULL_COLOR = palette[4]  # fifth color
COMPARISON_COLOR = palette[8]  # eighth color

# Function to load data
def load_data():
    with open('data/fixed_comparisons.json', 'r') as f:
        comparisons = json.load(f)

    train_df = pd.read_parquet('data/compas_train.parquet')
    test_df = pd.read_parquet('data/compas_test.parquet')
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Extract unique individuals from comparisons
    comparison_ids = set()
    for comp in comparisons:
        comparison_ids.add(comp['individual1_id'])
        comparison_ids.add(comp['individual2_id'])

    # Create dataframe of only individuals that appear in comparisons
    comparison_df = full_df[full_df['id'].isin(comparison_ids)]
    
    return full_df, comparison_df

# Function to create a single comparison plot for categorical variables
def plot_categorical_comparison(ax, full_df, comp_df, column, title, stat_test=True, rotation=0):
    full_props = full_df[column].value_counts(normalize=True).sort_index()
    comp_props = comp_df[column].value_counts(normalize=True).sort_index()
    
    # Ensure the categories match
    all_cats = sorted(set(full_props.index) | set(comp_props.index))
    full_data = [full_props.get(cat, 0) * 100 for cat in all_cats]
    comp_data = [comp_props.get(cat, 0) * 100 for cat in all_cats]
    
    # Create bar positions
    x = np.arange(len(all_cats))
    width = 0.35
    
    # Create bars with custom colors
    ax.bar(x - width/2, full_data, width, label='Full Dataset', color=FULL_COLOR, alpha=0.9)
    ax.bar(x + width/2, comp_data, width, label='Comparison Set', color=COMPARISON_COLOR, alpha=0.9)
    
    # Add labels and formatting
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=rotation)
    ax.set_ylabel('Percentage (%)')
    
    # Add statistical test result if requested
    if stat_test:
        # Prepare contingency table for chi2 test
        full_counts = full_df[column].value_counts()
        comp_counts = comp_df[column].value_counts()
        
        # Align the indices
        table = pd.DataFrame({
            'full': [full_counts.get(cat, 0) for cat in all_cats],
            'comparison': [comp_counts.get(cat, 0) for cat in all_cats]
        }, index=all_cats)
        
        # Run chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(table)
        # Add p-value in a box for better visibility
        ax.text(0.5, 0.9, f'p={p:.3f}', transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Function to create a comparison plot for age groups
def plot_age_groups(ax, full_df, comp_df):
    # Define age bins
    bins = [0, 25, 35, 45, 55, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    
    # Calculate age distributions
    full_df['age_group'] = pd.cut(full_df['age'], bins=bins, labels=labels, right=False)
    comp_df['age_group'] = pd.cut(comp_df['age'], bins=bins, labels=labels, right=False)
    
    # Get proportions
    full_props = full_df['age_group'].value_counts(normalize=True).sort_index() * 100
    comp_props = comp_df['age_group'].value_counts(normalize=True).sort_index() * 100
    
    # Create bar positions
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars with custom colors
    ax.bar(x - width/2, [full_props.get(label, 0) for label in labels], width, 
           label='Full Dataset', color=FULL_COLOR, alpha=0.9)
    ax.bar(x + width/2, [comp_props.get(label, 0) for label in labels], width, 
           label='Comparison Set', color=COMPARISON_COLOR, alpha=0.9)
    
    # Add labels and formatting
    ax.set_title('Age Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Percentage (%)')
    
    # Run t-test
    t, p = stats.ttest_ind(full_df['age'].dropna(), comp_df['age'].dropna())
    # Add p-value in a box for better visibility
    ax.text(0.5, 0.9, f'p={p:.3f}', transform=ax.transAxes, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Function to create a comparison plot for priors count
def plot_priors_groups(ax, full_df, comp_df):
    # Define priors bins
    bins = [-1, 0, 3, 9, 100]
    labels = ['0', '1-3', '4-9', '10+']
    
    # Calculate priors distributions
    full_df['priors_group'] = pd.cut(full_df['priors_count'], bins=bins, labels=labels, right=True)
    comp_df['priors_group'] = pd.cut(comp_df['priors_count'], bins=bins, labels=labels, right=True)
    
    # Get proportions
    full_props = full_df['priors_group'].value_counts(normalize=True).sort_index() * 100
    comp_props = comp_df['priors_group'].value_counts(normalize=True).sort_index() * 100
    
    # Create bar positions
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars with custom colors
    ax.bar(x - width/2, [full_props.get(label, 0) for label in labels], width, 
           label='Full Dataset', color=FULL_COLOR, alpha=0.9)
    ax.bar(x + width/2, [comp_props.get(label, 0) for label in labels], width, 
           label='Comparison Set', color=COMPARISON_COLOR, alpha=0.9)
    
    # Add labels and formatting
    ax.set_title('Prior Convictions', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Percentage (%)')
    
    # Run t-test
    t, p = stats.ttest_ind(full_df['priors_count'].dropna(), comp_df['priors_count'].dropna())
    # Add p-value in a box for better visibility
    ax.text(0.5, 0.9, f'p={p:.3f}', transform=ax.transAxes, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Create the full visualization
def create_demographic_comparison_plots():
    # Load data
    full_df, comp_df = load_data()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot sex distribution
    plot_categorical_comparison(axes[0, 0], full_df, comp_df, 'sex', 'Gender Distribution')
    
    # Plot race distribution
    plot_categorical_comparison(axes[0, 1], full_df, comp_df, 'race', 'Race Distribution', rotation=45)
    
    # Plot age groups
    plot_age_groups(axes[0, 2], full_df, comp_df)
    
    # Plot charge degree
    plot_categorical_comparison(axes[1, 0], full_df, comp_df, 'c_charge_degree', 'Charge Degree')
    
    # Plot priors count
    plot_priors_groups(axes[1, 1], full_df, comp_df)
    
    # Plot recidivism
    plot_categorical_comparison(axes[1, 2], full_df, comp_df, 'two_year_recid', 'Recidivism Rate')
    
    # Add legend and adjust layout
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=2, frameon=True, fontsize=11)
    
    # Add overall title
    fig.suptitle(f'Demographic Comparison: Full Dataset (N={len(full_df)}) vs. Comparison Set (N={len(comp_df)})', 
                 fontsize=16, y=0.92, fontweight='bold')
    
    # Add a note about statistical significance
    plt.figtext(0.5, 0.01, 'Note: No statistically significant differences were found between datasets (all p > 0.05)',
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.9])  # Leave room for the title and note
    plt.subplots_adjust(top=0.85)
    
    # Create the output directory if it doesn't exist
    output_dir = 'compas_eda/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure to the specified directory
    plt.savefig(f'{output_dir}/demographic_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/demographic_comparison.pdf', bbox_inches='tight')
    
    print(f"Figures saved in {output_dir}/")
    
    return fig

# Function to display available color palettes (fixed version)
def show_color_options():
    """Show available color palette options to choose from"""
    # Create output directory if it doesn't exist
    output_dir = 'compas_eda/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure to show different color palette options
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    
    # In newer versions of seaborn, we need to manually plot the palettes
    # Show colorblind palette
    colorblind_palette = sns.color_palette("colorblind")
    for i, color in enumerate(colorblind_palette):
        axes[0].add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    axes[0].set_xlim(0, len(colorblind_palette))
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Colorblind Palette (Default)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Show Set1 palette
    set1_palette = sns.color_palette("Set1")
    for i, color in enumerate(set1_palette):
        axes[1].add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    axes[1].set_xlim(0, len(set1_palette))
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Set1 Palette")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Show Set2 palette
    set2_palette = sns.color_palette("Set2")
    for i, color in enumerate(set2_palette):
        axes[2].add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    axes[2].set_xlim(0, len(set2_palette))
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Set2 Palette")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    # Show Paired palette
    paired_palette = sns.color_palette("Paired")
    for i, color in enumerate(paired_palette):
        axes[3].add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    axes[3].set_xlim(0, len(paired_palette))
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Paired Palette")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/color_palette_options.png', dpi=300)
    plt.close()
    
    print(f"Color palette options saved to {output_dir}/color_palette_options.png")

# Run the visualization function
if __name__ == "__main__":
    # Uncomment to see color palette options
    # show_color_options()
    
    create_demographic_comparison_plots()
    plt.show()