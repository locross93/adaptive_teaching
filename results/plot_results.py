import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for seaborn plots
sns.set_style("white")
sns.set_context("paper", font_scale=1.5)

# Set color palette
palette = sns.color_palette("Set1")

# Initialize Wandb API
api = wandb.Api()

# Function to get run data
def get_run_data(run_path):
    run = api.run(run_path)
    history = run.scan_history()
    return pd.DataFrame(history)

# List of runs you want to compare
# add generalizer
runs = [
    "stanford_autonomous_agent/adapt_fractions/g7kjvdal",
    "stanford_autonomous_agent/adapt_fractions/sdhhw9p5",
    "stanford_autonomous_agent/adapt_fractions/op28q56h",
    "stanford_autonomous_agent/pedagogy_lists/93d2693u",
]
# # multiply generalizer
# runs = [
#     "stanford_autonomous_agent/adapt_fractions/wxwwqtkc",
#     "stanford_autonomous_agent/adapt_fractions/y17cww02",
#     "stanford_autonomous_agent/adapt_fractions/s6zoxckb",
#     "stanford_autonomous_agent/adapt_fractions/93d2693u",
# ]

run_names = [
    "HM Agent",
    "ATOM",
    "GPT",
    "Random",
]

# Fetch data for each run
data = {}
end_stats = {}
for c, run_path in enumerate(runs):
    run_name = run_names[c]
    df = get_run_data(run_path)
    data[run_name] = df
    if 'end_stats/aucs/gold_prob' in df.columns:
        max_value = df['end_stats/aucs/gold_prob'].max()
        if pd.notna(max_value) and np.isfinite(max_value):
            end_stats[run_name] = max_value

# Plot the line graph
plt.figure(figsize=(12, 6))
for i, (run_name, df) in enumerate(data.items()):
    sns.lineplot(x='_step', y='student_learning/gold_total_prob', data=df, label=run_name, color=palette[i])

plt.xlabel('Step', fontsize=18)
plt.ylabel('Posterior of Target Concept', fontsize=18)
plt.title('Student Learning: Probability of Correct Concept', fontsize=20)
plt.legend(title='Model')
plt.grid(False)
plt.tight_layout()
plt.savefig('results/all_models_gold_total_probability.png', dpi=300)
plt.close()

# Create bar plot for end_stats/aucs/gold_prob
if end_stats:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(end_stats.keys()), y=list(end_stats.values()), palette=palette)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel('AUC of Target Concept Curve', fontsize=14)
    plt.title('AUC: Students’ Likelihood of Target Concept', fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    
    # Add value labels on top of each bar
    for i, v in enumerate(end_stats.values()):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('results/all_models_end_stats_auc_gold_prob.png', dpi=300)
    plt.close()

print("Plots have been saved in the 'results' directory.")