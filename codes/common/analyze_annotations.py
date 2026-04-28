#!/usr/bin/env python3
"""
Annotation Analysis Script - Optimized for Large Datasets
Analyzes annotation_analyze.csv and generates visualizations and summary reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from pathlib import Path
import warnings

try:
    import japanize_matplotlib  # noqa: F401
    _HAS_JAPANIZE = True
except Exception:
    _HAS_JAPANIZE = False

warnings.filterwarnings('ignore')


def configure_japanese_font() -> None:
    """Configure matplotlib to use an installed Japanese-capable font."""
    if _HAS_JAPANIZE:
        # seaborn theme setup can override font settings, so enforce again here.
        plt.rcParams['font.family'] = 'IPAexGothic'
        sans = plt.rcParams.get('font.sans-serif', [])
        if 'IPAexGothic' not in sans:
            plt.rcParams['font.sans-serif'] = ['IPAexGothic', *sans]
        print("Using Japanese font via japanize_matplotlib")
        plt.rcParams['axes.unicode_minus'] = False
        return

    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAGothic",
        "IPAexGothic",
        "TakaoGothic",
        "Yu Gothic",
        "Hiragino Sans",
        "MS Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    selected = None
    for name in candidates:
        if name in available:
            selected = name
            break

    if selected:
        plt.rcParams['font.family'] = selected
        print(f"Using Japanese font: {selected}")
    else:
        print("Warning: No Japanese font found. Labels may appear garbled.")

    # Avoid garbled minus signs with some CJK fonts.
    plt.rcParams['axes.unicode_minus'] = False


# Set style for better-looking plots
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 9
configure_japanese_font()

# Paths
CSV_PATH = Path(__file__).parent.parent / "reports" / "annotation_analyze.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading data efficiently...")
# Load only necessary columns for analysis
key_columns = [
    'video_stem', 'action_type', 'body_part', 'grip_or_contact', 'speed_or_force',
    'duration', 'motion_detail_char_len', 'target_object_char_len',
    'body_part_is_missing', 'action_type_is_missing', 'target_object_is_missing',
    'grip_or_contact_is_missing', 'speed_or_force_is_missing', 'posture_change_is_missing',
    'is_zero_duration', 'is_negative_duration', 'out_of_video', 'overlaps_prev', 'overlaps_next',
    'prompt_token_count', 'candidates_token_count', 'total_token_count',
    'coverage_ratio', 'motion_detail_has_number'
]

# Use dtype specification to reduce memory usage
dtypes = {
    'action_type': 'string',
    'body_part': 'string',
    'grip_or_contact': 'string',
    'speed_or_force': 'string',
}

df = pd.read_csv(CSV_PATH, usecols=key_columns, dtype=dtypes, low_memory=False)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

# ============================================================================
# BASIC STATISTICS
# ============================================================================
print("\n=== Basic Statistics ===")
basic_stats = {
    'total_rows': len(df),
    'total_columns': len(df.columns),
    'unique_videos': df['video_stem'].nunique() if 'video_stem' in df.columns else 0,
}

print(f"Total Rows: {basic_stats['total_rows']:,}")
print(f"Total Columns: {basic_stats['total_columns']}")
print(f"Unique Videos: {basic_stats['unique_videos']:,}")

# ============================================================================
# ACTION ANALYSIS
# ============================================================================
print("\n=== Action Analysis ===")

# Action type distribution
action_type_counts = df['action_type'].value_counts()
print(f"\nTop 10 Action Types:")
print(action_type_counts.head(10))

# Body part distribution
body_part_counts = df['body_part'].value_counts()
print(f"\nTop 10 Body Parts:")
print(body_part_counts.head(10))

# Grip or contact analysis
grip_contact_counts = df['grip_or_contact'].value_counts()
print(f"\nTop 10 Grip or Contact Types:")
print(grip_contact_counts.head(10))

# ============================================================================
# DURATION ANALYSIS
# ============================================================================
print("\n=== Duration Analysis ===")

duration_stats = {
    'mean_duration': df['duration'].mean(),
    'std_duration': df['duration'].std(),
    'min_duration': df['duration'].min(),
    'max_duration': df['duration'].max(),
    'median_duration': df['duration'].median(),
}

print(f"Mean Duration: {duration_stats['mean_duration']:.3f} sec")
print(f"Median Duration: {duration_stats['median_duration']:.3f} sec")
print(f"Std Duration: {duration_stats['std_duration']:.3f} sec")
print(f"Min Duration: {duration_stats['min_duration']:.3f} sec")
print(f"Max Duration: {duration_stats['max_duration']:.3f} sec")

# ============================================================================
# TEXT ANALYSIS
# ============================================================================
print("\n=== Text Analysis ===")

text_stats = {
    'mean_motion_detail_len': df['motion_detail_char_len'].mean(),
    'max_motion_detail_len': df['motion_detail_char_len'].max(),
    'min_motion_detail_len': df['motion_detail_char_len'].min(),
    'mean_target_object_len': df['target_object_char_len'].mean(),
    'max_target_object_len': df['target_object_char_len'].max(),
}

print(f"Motion Detail - Mean Length: {text_stats['mean_motion_detail_len']:.1f} chars")
print(f"Motion Detail - Max Length: {text_stats['max_motion_detail_len']} chars")
print(f"Target Object - Mean Length: {text_stats['mean_target_object_len']:.1f} chars")

# ============================================================================
# QUALITY ANALYSIS
# ============================================================================
print("\n=== Quality Analysis ===")

quality_stats = {
    'missing_body_part': df['body_part_is_missing'].sum(),
    'missing_action_type': df['action_type_is_missing'].sum(),
    'missing_target_object': df['target_object_is_missing'].sum(),
    'missing_grip_contact': df['grip_or_contact_is_missing'].sum(),
    'missing_posture_change': df['posture_change_is_missing'].sum(),
    'zero_duration': df['is_zero_duration'].sum(),
    'negative_duration': df['is_negative_duration'].sum(),
    'out_of_video': df['out_of_video'].sum(),
    'overlaps_prev': df['overlaps_prev'].sum(),
    'overlaps_next': df['overlaps_next'].sum(),
}

print(f"Missing Body Part: {quality_stats['missing_body_part']:,}")
print(f"Missing Action Type: {quality_stats['missing_action_type']:,}")
print(f"Missing Target Object: {quality_stats['missing_target_object']:,}")
print(f"Zero Duration: {quality_stats['zero_duration']:,}")
print(f"Out of Video: {quality_stats['out_of_video']:,}")
print(f"Overlaps with Previous: {quality_stats['overlaps_prev']:,}")
print(f"Overlaps with Next: {quality_stats['overlaps_next']:,}")

# ============================================================================
# TOKEN ANALYSIS (LLM Usage) - Optimized
# ============================================================================
print("\n=== Token Analysis ===")

# Use efficient aggregation
token_stats = {
    'total_prompt_tokens': df['prompt_token_count'].sum(),
    'total_completion_tokens': df['candidates_token_count'].sum(),
    'total_tokens': df['total_token_count'].sum(),
    'mean_prompt_tokens': df['prompt_token_count'].mean(),
    'mean_completion_tokens': df['candidates_token_count'].mean(),
}

print(f"Total Prompt Tokens: {token_stats['total_prompt_tokens']:,.0f}")
print(f"Total Completion Tokens: {token_stats['total_completion_tokens']:,.0f}")
print(f"Total Tokens: {token_stats['total_tokens']:,.0f}")
print(f"Mean Prompt Tokens per Action: {token_stats['mean_prompt_tokens']:.1f}")
print(f"Mean Completion Tokens per Action: {token_stats['mean_completion_tokens']:.1f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n=== Creating Visualizations ===")

fig = plt.figure(figsize=(18, 12))

# 1. Action Type Distribution (Top 15)
ax1 = plt.subplot(3, 3, 1)
top_actions = action_type_counts.head(15)
ax1.barh(range(len(top_actions)), top_actions.values, color='steelblue')
ax1.set_yticks(range(len(top_actions)))
ax1.set_yticklabels(top_actions.index, fontsize=8)
ax1.set_title('Top 15 Action Types', fontsize=11, fontweight='bold')
ax1.set_xlabel('Count')
ax1.invert_yaxis()

# 2. Body Part Distribution (Top 15)
ax2 = plt.subplot(3, 3, 2)
top_body_parts = body_part_counts.head(15)
ax2.barh(range(len(top_body_parts)), top_body_parts.values, color='coral')
ax2.set_yticks(range(len(top_body_parts)))
ax2.set_yticklabels(top_body_parts.index, fontsize=8)
ax2.set_title('Top 15 Body Parts', fontsize=11, fontweight='bold')
ax2.set_xlabel('Count')
ax2.invert_yaxis()

# 3. Grip or Contact Distribution (Top 10)
ax3 = plt.subplot(3, 3, 3)
top_grip = grip_contact_counts.head(10)
ax3.bar(range(len(top_grip)), top_grip.values, color='lightgreen')
ax3.set_xticks(range(len(top_grip)))
ax3.set_xticklabels(range(1, len(top_grip)+1), fontsize=8)
ax3.set_title('Top 10 Grip/Contact Types', fontsize=11, fontweight='bold')
ax3.set_ylabel('Count')

# 4. Duration Distribution (Sample for speed)
ax4 = plt.subplot(3, 3, 4)
duration_sample = df['duration'].dropna()
ax4.hist(duration_sample, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
ax4.set_title('Action Duration Distribution', fontsize=11, fontweight='bold')
ax4.set_xlabel('Duration (seconds)')
ax4.set_ylabel('Frequency')
ax4.set_xlim(0, duration_sample.quantile(0.99))

# 5. Motion Detail Length Distribution
ax5 = plt.subplot(3, 3, 5)
motion_len = df['motion_detail_char_len'].dropna()
ax5.hist(motion_len, bins=100, color='plum', edgecolor='black', alpha=0.7)
ax5.set_title('Motion Detail Length Distribution', fontsize=11, fontweight='bold')
ax5.set_xlabel('Character Count')
ax5.set_ylabel('Frequency')

# 6. Target Object Length Distribution
ax6 = plt.subplot(3, 3, 6)
target_len = df['target_object_char_len'].dropna()
ax6.hist(target_len, bins=100, color='lightsalmon', edgecolor='black', alpha=0.7)
ax6.set_title('Target Object Length Distribution', fontsize=11, fontweight='bold')
ax6.set_xlabel('Character Count')
ax6.set_ylabel('Frequency')

# 7. Quality Issues Summary
ax7 = plt.subplot(3, 3, 7)
quality_issues = {
    'Missing Body\nPart': quality_stats['missing_body_part'],
    'Missing Action\nType': quality_stats['missing_action_type'],
    'Missing Target\nObject': quality_stats['missing_target_object'],
    'Zero\nDuration': quality_stats['zero_duration'],
    'Out of\nVideo': quality_stats['out_of_video'],
    'Overlaps\nPrev': quality_stats['overlaps_prev'] / 1000,  # Divide for scale
}
quality_issues_filtered = {k: v for k, v in quality_issues.items() if v > 0}
if quality_issues_filtered:
    colors = ['tomato' if 'Overlaps' not in k else 'orange' for k in quality_issues_filtered.keys()]
    ax7.bar(range(len(quality_issues_filtered)), list(quality_issues_filtered.values()), color=colors)
    ax7.set_xticks(range(len(quality_issues_filtered)))
    ax7.set_xticklabels(quality_issues_filtered.keys(), fontsize=8)
    ax7.set_title('Data Quality Issues', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Count (Overlaps in 1000s)')
else:
    ax7.text(0.5, 0.5, 'No Quality Issues', ha='center', va='center', fontsize=11)
    ax7.set_title('Data Quality Issues', fontsize=11, fontweight='bold')
    ax7.axis('off')

# 8. Token Count Statistics
ax8 = plt.subplot(3, 3, 8)
token_stats_plot = {
    'Prompt\nTokens': token_stats['mean_prompt_tokens'],
    'Completion\nTokens': token_stats['mean_completion_tokens'],
    'Total\nTokens': df['total_token_count'].mean(),
}
ax8.bar(token_stats_plot.keys(), token_stats_plot.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax8.set_title('Mean Token Counts per Action', fontsize=11, fontweight='bold')
ax8.set_ylabel('Token Count')

# 9. Coverage Ratio Distribution
ax9 = plt.subplot(3, 3, 9)
coverage = df['coverage_ratio'].dropna()
ax9.hist(coverage, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
ax9.set_title('Coverage Ratio Distribution', fontsize=11, fontweight='bold')
ax9.set_xlabel('Coverage Ratio')
ax9.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'annotation_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to {OUTPUT_DIR / 'annotation_analysis.png'}")
plt.close()  # Free memory

# ============================================================================
# SUMMARY CSV
# ============================================================================
print("\n=== Creating Summary CSV ===")

summary_data = {
    'Metric': [
        # Basic Statistics
        'Total Rows',
        'Total Columns',
        'Unique Videos',
        
        # Duration Statistics
        'Mean Action Duration (sec)',
        'Median Action Duration (sec)',
        'Std Action Duration (sec)',
        'Min Action Duration (sec)',
        'Max Action Duration (sec)',
        
        # Text Statistics
        'Mean Motion Detail Length (chars)',
        'Max Motion Detail Length (chars)',
        'Mean Target Object Length (chars)',
        'Max Target Object Length (chars)',
        
        # Top Action Type & Body Part
        'Top Action Type',
        'Top Action Type Count',
        'Top Body Part',
        'Top Body Part Count',
        'Top Grip/Contact Type',
        'Top Grip/Contact Count',
        
        # Quality Metrics
        'Missing Body Parts',
        'Missing Action Types',
        'Missing Target Objects',
        'Zero Duration Actions',
        'Out of Video Actions',
        'Overlapping with Previous',
        'Overlapping with Next',
        
        # Token Statistics
        'Total Prompt Tokens',
        'Total Completion Tokens',
        'Total Tokens',
        'Mean Prompt Tokens per Action',
        'Mean Completion Tokens per Action',
        'Mean Total Tokens per Action',
        
        # Coverage
        'Mean Coverage Ratio',
        'Median Coverage Ratio',
    ],
    'Value': [
        # Basic Statistics
        f"{basic_stats['total_rows']:,}",
        f"{basic_stats['total_columns']}",
        f"{basic_stats['unique_videos']:,}",
        
        # Duration Statistics
        f"{duration_stats['mean_duration']:.4f}",
        f"{duration_stats['median_duration']:.4f}",
        f"{duration_stats['std_duration']:.4f}",
        f"{duration_stats['min_duration']:.4f}",
        f"{duration_stats['max_duration']:.4f}",
        
        # Text Statistics
        f"{text_stats['mean_motion_detail_len']:.1f}",
        f"{text_stats['max_motion_detail_len']}",
        f"{text_stats['mean_target_object_len']:.1f}",
        f"{text_stats['max_target_object_len']}",
        
        # Top Action Type & Body Part
        f"{action_type_counts.index[0]}",
        f"{action_type_counts.values[0]:,}",
        f"{body_part_counts.index[0]}",
        f"{body_part_counts.values[0]:,}",
        f"{grip_contact_counts.index[0]}",
        f"{grip_contact_counts.values[0]:,}",
        
        # Quality Metrics
        f"{quality_stats['missing_body_part']:,}",
        f"{quality_stats['missing_action_type']:,}",
        f"{quality_stats['missing_target_object']:,}",
        f"{quality_stats['zero_duration']:,}",
        f"{quality_stats['out_of_video']:,}",
        f"{quality_stats['overlaps_prev']:,}",
        f"{quality_stats['overlaps_next']:,}",
        
        # Token Statistics
        f"{token_stats['total_prompt_tokens']:,.0f}",
        f"{token_stats['total_completion_tokens']:,.0f}",
        f"{token_stats['total_tokens']:,.0f}",
        f"{token_stats['mean_prompt_tokens']:.1f}",
        f"{token_stats['mean_completion_tokens']:.1f}",
        f"{df['total_token_count'].mean():.1f}",
        
        # Coverage
        f"{df['coverage_ratio'].mean():.4f}",
        f"{df['coverage_ratio'].median():.4f}",
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_csv_path = OUTPUT_DIR / 'summary_statistics.csv'
summary_df.to_csv(summary_csv_path, index=False)
print(f"✓ Saved summary CSV to {summary_csv_path}")

# ============================================================================
# DETAILED DISTRIBUTION CSVS
# ============================================================================
print("\n=== Creating Detailed Distribution CSVs ===")

# Action Type Distribution (Top 50)
action_dist_df = pd.DataFrame({
    'Action Type': action_type_counts.head(50).index,
    'Count': action_type_counts.head(50).values,
    'Percentage': (action_type_counts.head(50).values / action_type_counts.sum() * 100).round(2)
})
action_dist_df.to_csv(OUTPUT_DIR / 'action_type_distribution.csv', index=False)
print(f"✓ Saved action type distribution to {OUTPUT_DIR / 'action_type_distribution.csv'}")

# Body Part Distribution (Top 50)
body_part_dist_df = pd.DataFrame({
    'Body Part': body_part_counts.head(50).index,
    'Count': body_part_counts.head(50).values,
    'Percentage': (body_part_counts.head(50).values / body_part_counts.sum() * 100).round(2)
})
body_part_dist_df.to_csv(OUTPUT_DIR / 'body_part_distribution.csv', index=False)
print(f"✓ Saved body part distribution to {OUTPUT_DIR / 'body_part_distribution.csv'}")

# Grip/Contact Distribution (Top 50)
grip_dist_df = pd.DataFrame({
    'Grip/Contact Type': grip_contact_counts.head(50).index,
    'Count': grip_contact_counts.head(50).values,
    'Percentage': (grip_contact_counts.head(50).values / grip_contact_counts.sum() * 100).round(2)
})
grip_dist_df.to_csv(OUTPUT_DIR / 'grip_contact_distribution.csv', index=False)
print(f"✓ Saved grip/contact distribution to {OUTPUT_DIR / 'grip_contact_distribution.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("Generated files:")
print(f"  - annotation_analysis.png (visualization)")
print(f"  - summary_statistics.csv")
print(f"  - action_type_distribution.csv")
print(f"  - body_part_distribution.csv")
print(f"  - grip_contact_distribution.csv")

# Clean up
del df
print("\n✓ Memory cleaned up")
