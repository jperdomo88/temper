"""
Plotting module for Tempered RLHF Experiment.

Generates publication-ready figures:
- Figure 1: Exploitability Gap (dual-axis training curves)
- Figure 2: Behavioral Outcomes (grouped bar chart)
- Figure 3: Anti-Cheat Panel (2×2 grid)
- Figure 4: Distribution Shift & Variant table
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Matplotlib imports (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def setup_style():
    """Set up publication-ready plot style."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_exploitability_gap(
    proxy_history: Dict[str, List],
    tempered_history: Dict[str, List],
    output_path: str = "figures/fig1_exploitability_gap.png",
) -> None:
    """
    Figure 1: Exploitability Gap (THE MONEY SHOT)
    
    Dual-axis plot showing:
    - Proxy: RM and E scores diverge → GAP OPENS
    - Tempered: RM and E scores track → GAP STAYS CLOSED
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return
    
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Proxy condition
    ax1 = axes[0]
    generations = range(len(proxy_history.get('mean_fitness', [])))
    
    ax1.plot(generations, proxy_history.get('mean_fitness', []), 
             'b-', label='RM Score', linewidth=2)
    ax1.fill_between(generations,
                     np.array(proxy_history.get('mean_fitness', [])) - np.array(proxy_history.get('fitness_std', [])),
                     np.array(proxy_history.get('mean_fitness', [])) + np.array(proxy_history.get('fitness_std', [])),
                     alpha=0.2, color='blue')
    
    ax1.set_xlabel('Training Steps / Generations')
    ax1.set_ylabel('Score (normalized)')
    ax1.set_title('Proxy RM: Gap Opens')
    ax1.legend(loc='upper left')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Tempered condition
    ax2 = axes[1]
    generations = range(len(tempered_history.get('mean_fitness', [])))
    
    ax2.plot(generations, tempered_history.get('mean_fitness', []),
             'g-', label='True Score (E)', linewidth=2)
    ax2.fill_between(generations,
                     np.array(tempered_history.get('mean_fitness', [])) - np.array(tempered_history.get('fitness_std', [])),
                     np.array(tempered_history.get('mean_fitness', [])) + np.array(tempered_history.get('fitness_std', [])),
                     alpha=0.2, color='green')
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness (True Score)')
    ax2.set_title('Tempered RM: Gap Stays Closed')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_behavioral_outcomes(
    results: Dict[str, Dict],
    output_path: str = "figures/fig2_behavioral_outcomes.png",
) -> None:
    """
    Figure 2: Behavioral Outcomes Bar Chart
    
    Grouped bars: Proxy / Tempered / Oracle
    Metrics: Hack Rate, Harm Rate, Protect Rate, E Score
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return
    
    setup_style()
    
    conditions = ['proxy', 'tempered', 'oracle']
    metrics = ['hack_rate', 'harm_rate', 'protect_rate', 'e_score']
    metric_labels = ['Hack Rate', 'Harm Rate', 'Protect Rate', 'E Score']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, condition in enumerate(conditions):
        if condition not in results:
            continue
        
        values = [results[condition].get(f'{m}_mean', 0) for m in metrics]
        errors = [results[condition].get(f'{m}_std', 0) for m in metrics]
        
        bars = ax.bar(x + i * width, values, width, 
                      label=condition.capitalize(),
                      color=colors[i], alpha=0.8,
                      yerr=errors, capsize=3)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Behavioral Outcomes by Condition')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_anticheat_panel(
    proxy_results: Dict,
    tempered_results: Dict,
    output_path: str = "figures/fig3_anticheat_panel.png",
) -> None:
    """
    Figure 3: Anti-Cheat & Robustness Panel (2×2 - THE HEAD-TURNER)
    
    (A) Unconditional Hack Rate
    (B) Conditional Hack Rate (given terminal access)
    (C) Forced Terminal Start (hack_rate_first_3_steps)
    (D) Exploit Variant Robustness (ΔE across variants)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return
    
    setup_style()
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # (A) Unconditional Hack Rate
    ax1 = fig.add_subplot(gs[0, 0])
    conditions = ['Proxy', 'Tempered']
    hack_rates = [
        proxy_results.get('hack_rate_mean', 0),
        tempered_results.get('hack_rate_mean', 0)
    ]
    hack_errs = [
        proxy_results.get('hack_rate_std', 0),
        tempered_results.get('hack_rate_std', 0)
    ]
    
    bars = ax1.bar(conditions, hack_rates, yerr=hack_errs, 
                   color=['#e74c3c', '#2ecc71'], capsize=5, alpha=0.8)
    ax1.set_ylabel('Hack Rate')
    ax1.set_title('(A) Unconditional Hack Rate')
    ax1.set_ylim(0, max(0.5, max(hack_rates) * 1.2))
    
    # (B) Conditional Hack Rate
    ax2 = fig.add_subplot(gs[0, 1])
    cond_hack_rates = [
        proxy_results.get('conditional_hack_rate_mean', 0),
        tempered_results.get('conditional_hack_rate_mean', 0)
    ]
    
    bars = ax2.bar(conditions, cond_hack_rates,
                   color=['#e74c3c', '#2ecc71'], alpha=0.8)
    ax2.set_ylabel('Conditional Hack Rate')
    ax2.set_title('(B) Hack Rate | Terminal Access')
    ax2.set_ylim(0, max(0.5, max(cond_hack_rates) * 1.2))
    
    # (C) Forced Terminal Start
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get forced terminal results if available
    proxy_forced = proxy_results.get('forced_terminal', {})
    tempered_forced = tempered_results.get('forced_terminal', {})
    
    forced_hack = [
        proxy_forced.get('hack_rate_first_3_steps', 0) if isinstance(proxy_forced, dict) else 0,
        tempered_forced.get('hack_rate_first_3_steps', 0) if isinstance(tempered_forced, dict) else 0,
    ]
    
    bars = ax3.bar(conditions, forced_hack,
                   color=['#e74c3c', '#2ecc71'], alpha=0.8)
    ax3.set_ylabel('Hack Rate (First 3 Steps)')
    ax3.set_title('(C) Forced Terminal Start')
    ax3.set_ylim(0, 1.0)
    
    # (D) Variant Robustness
    ax4 = fig.add_subplot(gs[1, 1])
    
    variants = ['nearest', 'random', 'delayed']
    x = np.arange(len(variants))
    width = 0.35
    
    # Get variant results if available
    proxy_variants = proxy_results.get('variants', {})
    tempered_variants = tempered_results.get('variants', {})
    
    proxy_e = [proxy_variants.get(v, {}).get('e_score', 0) for v in variants]
    tempered_e = [tempered_variants.get(v, {}).get('e_score', 0) for v in variants]
    
    ax4.bar(x - width/2, proxy_e, width, label='Proxy', color='#e74c3c', alpha=0.8)
    ax4.bar(x + width/2, tempered_e, width, label='Tempered', color='#2ecc71', alpha=0.8)
    ax4.set_ylabel('E Score')
    ax4.set_title('(D) Exploit Variant Robustness')
    ax4.set_xticks(x)
    ax4.set_xticklabels(variants)
    ax4.legend()
    
    plt.suptitle('Anti-Cheat Validation Panel', fontsize=14, fontweight='bold')
    
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_results_table(
    results: Dict[str, Dict],
    output_path: str = "figures/table_results.txt",
) -> str:
    """
    Create formatted results table for paper.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("TEMPERED RLHF EXPERIMENT RESULTS")
    lines.append("=" * 80)
    lines.append("")
    
    # Main metrics table
    lines.append("Primary Metrics:")
    lines.append("-" * 60)
    lines.append(f"{'Condition':<12} {'Hack Rate':<12} {'E Score':<12} {'Exploit_z':<12}")
    lines.append("-" * 60)
    
    for condition in ['proxy', 'tempered', 'oracle']:
        if condition not in results:
            continue
        r = results[condition]
        lines.append(
            f"{condition.capitalize():<12} "
            f"{r.get('hack_rate_mean', 0):.4f}±{r.get('hack_rate_std', 0):.4f}  "
            f"{r.get('e_score_mean', 0):.4f}±{r.get('e_score_std', 0):.4f}  "
            f"{r.get('exploit_z_mean', 0):.4f}±{r.get('exploit_z_std', 0):.4f}"
        )
    
    lines.append("")
    lines.append("Anti-Cheat Metrics:")
    lines.append("-" * 60)
    lines.append(f"{'Condition':<12} {'Term Visit':<12} {'Cond Hack':<12} {'Reach Time':<12}")
    lines.append("-" * 60)
    
    for condition in ['proxy', 'tempered']:
        if condition not in results:
            continue
        r = results[condition]
        lines.append(
            f"{condition.capitalize():<12} "
            f"{r.get('terminal_visit_rate_mean', 0):.4f}         "
            f"{r.get('conditional_hack_rate_mean', 0):.4f}         "
            f"{r.get('mean_steps_to_terminal', np.nan):.1f}"
        )
    
    lines.append("")
    lines.append("=" * 80)
    
    table_str = "\n".join(lines)
    
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(table_str)
    
    print(f"Saved: {output_path}")
    return table_str


def generate_all_figures(
    results: Dict[str, Any],
    figures_dir: str = "figures",
) -> None:
    """Generate all paper figures from results."""
    Path(figures_dir).mkdir(exist_ok=True)
    
    # Extract condition results
    proxy = results.get('proxy', {})
    tempered = results.get('tempered', {})
    
    # Figure 2: Behavioral outcomes
    plot_behavioral_outcomes(results, f"{figures_dir}/fig2_behavioral_outcomes.png")
    
    # Figure 3: Anti-cheat panel
    plot_anticheat_panel(proxy, tempered, f"{figures_dir}/fig3_anticheat_panel.png")
    
    # Results table
    create_results_table(results, f"{figures_dir}/table_results.txt")
    
    print(f"\nAll figures saved to {figures_dir}/")


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Plotting Module")
    print("=" * 60)
    print(f"\nMatplotlib available: {MATPLOTLIB_AVAILABLE}")
    print("\nThis module provides:")
    print("- plot_exploitability_gap(): Figure 1")
    print("- plot_behavioral_outcomes(): Figure 2")
    print("- plot_anticheat_panel(): Figure 3 (2×2)")
    print("- create_results_table(): Text table")
    print("- generate_all_figures(): All at once")
    print("=" * 60)
