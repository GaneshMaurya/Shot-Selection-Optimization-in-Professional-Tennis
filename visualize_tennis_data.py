 
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Load the analysis results
with open('analysis_results/basic_stats.json', 'r') as f:
    basic_stats = json.load(f)
    
with open('analysis_results/shot_stats.json', 'r') as f:
    shot_stats = json.load(f)
    
with open('analysis_results/outcome_stats.json', 'r') as f:
    outcome_stats = json.load(f)
    
with open('analysis_results/pattern_stats.json', 'r') as f:
    pattern_stats = json.load(f)

print("Generating visualizations...")

# Set up the style
plt.style.use('ggplot')
sns.set_palette("muted")

# 1. Basic Dataset Statistics
print("Creating basic statistics visualizations...")

# Tournament distribution
if len(basic_stats['tournamentTypes']) > 0:
    plt.figure(figsize=(12, 6))
    tournaments = list(basic_stats['tournamentTypes'].keys())
    counts = list(basic_stats['tournamentTypes'].values())
    
    # Sort for better visualization
    sorted_data = sorted(zip(tournaments, counts), key=lambda x: x[1], reverse=True)
    tournaments, counts = zip(*sorted_data) if sorted_data else ([], [])
    
    plt.bar(tournaments, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Tournament Distribution')
    plt.xlabel('Tournament')
    plt.ylabel('Number of Matches')
    plt.tight_layout()
    plt.savefig('visualizations/tournament_distribution.png')
    plt.close()

# Points per match distribution
if 'pointsPerMatch' in basic_stats and 'distribution' in basic_stats['pointsPerMatch']:
    plt.figure(figsize=(10, 6))
    
    # Convert dictionary to lists for plotting
    points_per_match = {int(k): v for k, v in basic_stats['pointsPerMatch']['distribution'].items()}
    points = list(points_per_match.keys())
    frequencies = list(points_per_match.values())
    
    plt.bar(points, frequencies)
    plt.title('Points per Match Distribution')
    plt.xlabel('Number of Points')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/points_per_match.png')
    plt.close()

# 2. Shot Distribution Visualizations
print("Creating shot distribution visualizations...")

# Shot types distribution
if len(shot_stats['shotTypes']) > 0:
    plt.figure(figsize=(12, 6))
    shot_types = list(shot_stats['shotTypes'].keys())
    shot_counts = list(shot_stats['shotTypes'].values())
    
    # Sort by frequency
    sorted_data = sorted(zip(shot_types, shot_counts), key=lambda x: x[1], reverse=True)
    shot_types, shot_counts = zip(*sorted_data) if sorted_data else ([], [])
    
    plt.bar(shot_types, shot_counts)
    plt.title('Shot Type Distribution')
    plt.xlabel('Shot Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/shot_types.png')
    plt.close()

# Serve types distribution
plt.figure(figsize=(12, 6))
serve_types = ['wide', 'body', 'T', 'other']
first_serve_counts = [shot_stats['serveTypes']['first'].get(t, 0) for t in serve_types]
second_serve_counts = [shot_stats['serveTypes']['second'].get(t, 0) for t in serve_types]

# Only include serve types that have data
valid_indices = [i for i, (first, second) in enumerate(zip(first_serve_counts, second_serve_counts)) 
                 if first > 0 or second > 0]
serve_types = [serve_types[i] for i in valid_indices]
first_serve_counts = [first_serve_counts[i] for i in valid_indices]
second_serve_counts = [second_serve_counts[i] for i in valid_indices]

if len(serve_types) > 0:
    x = np.arange(len(serve_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, first_serve_counts, width, label='First Serve')
    ax.bar(x + width/2, second_serve_counts, width, label='Second Serve')
    
    ax.set_xticks(x)
    ax.set_xticklabels(serve_types)
    ax.set_title('Serve Type Distribution')
    ax.legend()
    ax.set_xlabel('Serve Type')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/serve_types.png')
    plt.close()

# Rally length distribution
if 'rallyLengths' in shot_stats and 'distribution' in shot_stats['rallyLengths']:
    plt.figure(figsize=(12, 6))
    
    # Convert dictionary to lists for plotting
    rally_lengths = {int(k): v for k, v in shot_stats['rallyLengths']['distribution'].items()}
    lengths = list(rally_lengths.keys())
    frequencies = list(rally_lengths.values())
    
    if lengths:  # Check if there's data to plot
        plt.bar(lengths, frequencies)
        plt.title('Rally Length Distribution')
        plt.xlabel('Rally Length (shots)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/rally_lengths.png')
    plt.close()

# Shot directions pie chart
if 'shotDirections' in shot_stats:
    plt.figure(figsize=(8, 8))
    directions = ['to Right', 'to Middle', 'to Left']
    values = [shot_stats['shotDirections'].get('1', 0),
              shot_stats['shotDirections'].get('2', 0),
              shot_stats['shotDirections'].get('3', 0)]
    
    if sum(values) > 0:  # Only create pie chart if there's data
        plt.pie(values, labels=directions, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Shot Direction Distribution')
        plt.savefig('visualizations/shot_directions.png')
    plt.close()

# 3. Outcome Analysis Visualizations
print("Creating outcome analysis visualizations...")

# Point outcomes
if 'outcomes' in outcome_stats and len(outcome_stats['outcomes']) > 0:
    plt.figure(figsize=(12, 6))
    outcome_types = list(outcome_stats['outcomes'].keys())
    outcome_counts = list(outcome_stats['outcomes'].values())
    
    # Sort for better visualization
    sorted_data = sorted(zip(outcome_types, outcome_counts), key=lambda x: x[1], reverse=True)
    outcome_types, outcome_counts = zip(*sorted_data) if sorted_data else ([], [])
    
    plt.bar(outcome_types, outcome_counts)
    plt.title('Point Outcome Distribution')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/outcome_distribution.png')
    plt.close()

# Rally length at point end
if 'pointsEndedAtRallyLength' in outcome_stats:
    plt.figure(figsize=(12, 6))
    
    # Convert dictionary to lists for plotting
    rally_end_lengths = {int(k): v for k, v in outcome_stats['pointsEndedAtRallyLength'].items()}
    lengths = list(rally_end_lengths.keys())
    counts = list(rally_end_lengths.values())
    
    if lengths:  # Check if there's data to plot
        plt.bar(lengths, counts)
        plt.title('Points Ended by Rally Length')
        plt.xlabel('Rally Length (shots)')
        plt.ylabel('Number of Points')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/points_ended_by_rally_length.png')
    plt.close()

# Outcome by shot type
if 'outcomeByShotType' in outcome_stats and len(outcome_stats['outcomeByShotType']) > 0:
    # Prepare data for stacked bar chart
    shot_types = list(outcome_stats['outcomeByShotType'].keys())
    
    # Get all unique outcomes
    all_outcomes = set()
    for shot_type, outcomes in outcome_stats['outcomeByShotType'].items():
        all_outcomes.update(outcomes.keys())
    all_outcomes = list(all_outcomes)
    
    # Create matrix of values
    data = []
    for outcome in all_outcomes:
        row = []
        for shot in shot_types:
            row.append(outcome_stats['outcomeByShotType'][shot].get(outcome, 0))
        data.append(row)
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(shot_types))
    
    for i, outcome in enumerate(all_outcomes):
        ax.bar(shot_types, data[i], bottom=bottom, label=outcome)
        bottom += data[i]
    
    ax.set_title('Outcome by Shot Type')
    ax.set_xlabel('Shot Type')
    ax.set_ylabel('Count')
    ax.set_xticklabels(shot_types, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig('visualizations/outcome_by_shot_type.png')
    plt.close()

# 4. Pattern Analysis Visualizations
print("Creating pattern analysis visualizations...")

# Top serve+1 patterns
if 'topServePlusOnePatterns' in pattern_stats and len(pattern_stats['topServePlusOnePatterns']) > 0:
    plt.figure(figsize=(12, 8))
    patterns = list(pattern_stats['topServePlusOnePatterns'].keys())
    counts = list(pattern_stats['topServePlusOnePatterns'].values())
    
    # Create legend mapping for patterns
    pattern_legend = {
        '4': 'Wide serve', 
        '5': 'Body serve', 
        '6': 'T serve',
        'f': 'Forehand', 
        'b': 'Backhand', 
        's': 'Backhand slice'
    }
    
    # Create readable labels
    labels = []
    for pattern in patterns:
        if '-' in pattern:
            serve, return_shot = pattern.split('-')
            serve_desc = pattern_legend.get(serve, serve)
            return_desc = pattern_legend.get(return_shot, return_shot)
            labels.append(f"{serve_desc} → {return_desc}")
        else:
            labels.append(pattern)
    
    plt.barh(labels, counts)  # Using horizontal bars for better label display
    plt.title('Top Serve+1 Patterns')
    plt.xlabel('Count')
    plt.ylabel('Pattern')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/top_serve_plus_one_patterns.png')
    plt.close()

# Pattern win rates
if 'patternWinRates' in pattern_stats and len(pattern_stats['patternWinRates']) > 0:
    # Extract patterns with sufficient data
    patterns = []
    win_rates = []
    totals = []
    
    for pattern, data in pattern_stats['patternWinRates'].items():
        if data['total'] >= 5:  # Only include patterns with sufficient sample size
            if '-' in pattern:
                serve, return_shot = pattern.split('-')
                serve_desc = pattern_legend.get(serve, serve)
                return_desc = pattern_legend.get(return_shot, return_shot)
                patterns.append(f"{serve_desc} → {return_desc}")
            else:
                patterns.append(pattern)
            
            win_rates.append(float(data['winRate'].strip('%')))
            totals.append(data['total'])
    
    if patterns:  # Check if we have data to plot
        # Sort by win rate for better visualization
        sorted_data = sorted(zip(patterns, win_rates, totals), key=lambda x: x[1], reverse=True)
        patterns, win_rates, totals = zip(*sorted_data)
        
        # Only show top 15
        patterns = patterns[:15]
        win_rates = win_rates[:15]
        totals = totals[:15]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot win rates in the first subplot
        bars = ax1.barh(patterns, win_rates, color='skyblue')
        ax1.set_title('Server Win Rate by Serve+1 Pattern')
        ax1.set_xlabel('Win Rate (%)')
        ax1.set_ylabel('Pattern')
        ax1.set_xlim(0, 100)
        
        # Add percentage labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 2, bar.get_y() + bar.get_height()/2, 
                    f"{win_rates[i]:.1f}%", 
                    va='center')
        
        # Plot sample sizes in the second subplot
        ax2.barh(patterns, totals, color='lightgreen')
        ax2.set_title('Sample Size by Serve+1 Pattern')
        ax2.set_xlabel('Number of Points')
        ax2.set_ylabel('')  # No need for y-label as it's shared
        
        plt.tight_layout()
        plt.savefig('visualizations/pattern_win_rates.png')
        plt.close()

# Top rally sequences
if 'topRallySequences' in pattern_stats and len(pattern_stats['topRallySequences']) > 0:
    plt.figure(figsize=(12, 8))
    sequences = list(pattern_stats['topRallySequences'].keys())
    counts = list(pattern_stats['topRallySequences'].values())
    
    # Create more readable labels for sequences
    shot_legend = {
        'f': 'FH', 
        'b': 'BH', 
        's': 'BH slice',
        'r': 'FH slice',
        'v': 'FH volley',
        'z': 'BH volley',
        'o': 'Overhead',
        'l': 'FH lob',
        'm': 'BH lob'
    }
    
    labels = []
    for seq in sequences:
        if len(seq) >= 5:  # If it has the expected format with dashes
            parts = seq.split('-')
            readable = ' → '.join([shot_legend.get(p, p) for p in parts])
            labels.append(readable)
        else:
            labels.append(seq)
    
    plt.barh(labels, counts)  # Using horizontal bars for better label display
    plt.title('Top Rally Sequences')
    plt.xlabel('Count')
    plt.ylabel('Sequence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/top_rally_sequences.png')
    plt.close()

# 5. Create Serve Plus One Effectiveness Analysis
if ('outcomeByServePlusOne' in pattern_stats and 
    len(pattern_stats['outcomeByServePlusOne']) > 0 and 
    'topServePlusOnePatterns' in pattern_stats):
    
    # Get top patterns by frequency
    top_patterns = list(pattern_stats['topServePlusOnePatterns'].keys())[:8]  # Limit to top 8 for readability
    
    # Determine outcome categories
    all_outcomes = set()
    for pattern in top_patterns:
        if pattern in pattern_stats['outcomeByServePlusOne']:
            all_outcomes.update(pattern_stats['outcomeByServePlusOne'][pattern].keys())
    
    # Group into major categories for clearer visualization
    outcome_groups = {
        'Winner': ['Winner', 'Ace', 'Serve Winner'],
        'Forced Error': ['Forced Error'],
        'Unforced Error': ['Unforced Error', 'Double Fault'],
        'Other': ['unknown']
    }
    
    # Create matrix for heatmap
    matrix_data = []
    pattern_labels = []
    
    for pattern in top_patterns:
        if pattern in pattern_stats['outcomeByServePlusOne']:
            outcomes = pattern_stats['outcomeByServePlusOne'][pattern]
            
            # Create friendly pattern label
            if '-' in pattern:
                serve, return_shot = pattern.split('-')
                serve_desc = pattern_legend.get(serve, serve)
                return_desc = pattern_legend.get(return_shot, return_shot)
                pattern_label = f"{serve_desc} → {return_desc}"
            else:
                pattern_label = pattern
            
            pattern_labels.append(pattern_label)
            
            # Count outcomes by category
            grouped_outcomes = []
            for group_name, outcome_list in outcome_groups.items():
                count = sum(outcomes.get(outcome, 0) for outcome in outcome_list)
                grouped_outcomes.append(count)
            
            matrix_data.append(grouped_outcomes)
    
    if matrix_data:  # Only create visualization if we have data
        # Convert to numpy array
        matrix_data = np.array(matrix_data)
        
        # Convert to percentages for each row
        row_sums = matrix_data.sum(axis=1)
        matrix_pct = np.zeros_like(matrix_data, dtype=float)
        for i, total in enumerate(row_sums):
            if total > 0:
                matrix_pct[i] = matrix_data[i] / total * 100
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(matrix_pct, annot=True, fmt='.1f', 
                         xticklabels=list(outcome_groups.keys()),
                         yticklabels=pattern_labels,
                         cmap='YlGnBu')
        ax.set_title('Outcome Distribution by Serve+1 Pattern (%)')
        plt.tight_layout()
        plt.savefig('visualizations/serve_plus_one_outcome_heatmap.png')
        plt.close()

print("All visualizations generated successfully!")
