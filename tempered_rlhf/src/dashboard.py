"""
Live Dashboard for Tempered RLHF Experiment.

Real-time terminal visualization using Rich library.
Shows training progress, metrics, and alerts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Rich imports (optional - graceful fallback if not available)
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ExperimentDashboard:
    """
    Live terminal dashboard showing experiment progress.
    Updates in real-time as training proceeds.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # State
        self.current_condition = ""
        self.current_seed = 0
        self.total_seeds = 1
        self.current_generation = 0
        self.total_generations = 1
        self.current_rm = 0
        self.pop_size = 1
        self.current_step = 0
        self.total_steps = 1
        self.laundering_mode = False
        
        # Metrics
        self.hack_rate = 0.0
        self.e_score = 0.0
        self.exploit_z = 0.0
        self.fitness = 0.0
        self.laundering_rate = 0.0
        
        # History for trends
        self.hack_rate_history = []
        self.e_score_history = []
        self.exploit_z_history = []
        self.fitness_history = []
        self.laundering_history = []
        
        # Best values
        self.best_hack_rate = float('inf')
        self.best_e_score = float('-inf')
        self.best_exploit_z = float('inf')
        self.best_fitness = float('-inf')
        self.best_laundering_rate = float('inf')
        
        self.initial_fitness = None
        self.metrics_history = []
        self.alerts = []
        
        if RICH_AVAILABLE:
            self.console = Console()
    
    def update(self, **kwargs) -> None:
        """Update dashboard with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
                # Update history
                history_key = f"{key}_history"
                if hasattr(self, history_key):
                    getattr(self, history_key).append(value)
                
                # Update best values
                best_key = f"best_{key}"
                if hasattr(self, best_key):
                    current_best = getattr(self, best_key)
                    # Lower is better for hack_rate, exploit_z, laundering_rate
                    if key in ['hack_rate', 'exploit_z', 'laundering_rate']:
                        if value < current_best:
                            setattr(self, best_key, value)
                    else:
                        if value > current_best:
                            setattr(self, best_key, value)
        
        # Track initial fitness
        if self.initial_fitness is None and 'fitness' in kwargs:
            self.initial_fitness = kwargs['fitness']
    
    def _trend(self, history: List[float], window: int = 5) -> str:
        """Compute trend indicator."""
        if len(history) < 2:
            return "—"
        recent = history[-window:]
        if len(recent) < 2:
            return "—"
        delta = recent[-1] - recent[0]
        if delta > 0.01:
            return "↑"
        elif delta < -0.01:
            return "↓"
        return "→"
    
    def _gen_bar(self, current: int, total: int, width: int = 20) -> str:
        """Generate progress bar string."""
        if total == 0:
            return "░" * width
        filled = int(width * current / total)
        return "█" * filled + "░" * (width - filled)
    
    def render_simple(self) -> str:
        """Simple text-based progress (no Rich)."""
        return (
            f"\r[{self.current_condition}] "
            f"Seed {self.current_seed}/{self.total_seeds} | "
            f"Gen {self.current_generation}/{self.total_generations} | "
            f"RM {self.current_rm}/{self.pop_size} | "
            f"Step {self.current_step}/{self.total_steps} | "
            f"Hack: {self.hack_rate:.3f} | "
            f"E: {self.e_score:.3f}"
        )
    
    def print_simple(self) -> None:
        """Print simple progress line."""
        print(self.render_simple(), end="", flush=True)
    
    def create_layout(self) -> 'Layout':
        """Create Rich layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5)
        )
        layout["body"].split_row(
            Layout(name="progress", ratio=1),
            Layout(name="metrics", ratio=2)
        )
        return layout
    
    def render_header(self) -> 'Panel':
        """Render header panel."""
        return Panel(
            f"[bold cyan]TEMPERED RLHF EXPERIMENT[/] | "
            f"Condition: [yellow]{self.current_condition}[/] | "
            f"Seed: [green]{self.current_seed}/{self.total_seeds}[/]",
            style="blue"
        )
    
    def render_progress(self) -> 'Panel':
        """Render progress panel."""
        table = Table(title="Training Progress", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Progress", style="green")
        table.add_column("Bar", style="yellow")
        
        if self.current_condition == "tempered":
            table.add_row(
                "Generation",
                f"{self.current_generation}/{self.total_generations}",
                self._gen_bar(self.current_generation, self.total_generations)
            )
            table.add_row(
                "RM in Gen",
                f"{self.current_rm}/{self.pop_size}",
                self._gen_bar(self.current_rm, self.pop_size)
            )
        
        table.add_row(
            "Policy Steps",
            f"{self.current_step}/{self.total_steps}",
            self._gen_bar(self.current_step, self.total_steps)
        )
        
        return Panel(table)
    
    def render_metrics(self) -> 'Panel':
        """Render metrics panel."""
        table = Table(title="Live Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="white")
        table.add_column("Best", style="green")
        table.add_column("Trend", style="yellow")
        
        metrics = [
            ("Hack Rate", self.hack_rate, self.best_hack_rate, 
             self._trend(self.hack_rate_history)),
            ("E Score", self.e_score, self.best_e_score, 
             self._trend(self.e_score_history)),
            ("Exploit_z", self.exploit_z, self.best_exploit_z, 
             self._trend(self.exploit_z_history)),
            ("Fitness", self.fitness, self.best_fitness, 
             self._trend(self.fitness_history)),
        ]
        
        for name, current, best, trend in metrics:
            # Format best appropriately
            if name == "Hack Rate":
                best_str = f"{best:.4f}" if best < float('inf') else "—"
            elif name in ["E Score", "Fitness"]:
                best_str = f"{best:.4f}" if best > float('-inf') else "—"
            else:
                best_str = f"{best:.4f}" if best < float('inf') else "—"
            
            table.add_row(name, f"{current:.4f}", best_str, trend)
        
        if self.laundering_mode:
            table.add_row(
                "Laundering",
                f"{self.laundering_rate:.4f}",
                f"{self.best_laundering_rate:.4f}" if self.best_laundering_rate < float('inf') else "—",
                self._trend(self.laundering_history)
            )
        
        return Panel(table)
    
    def render_footer(self) -> 'Panel':
        """Render alerts panel."""
        alerts = []
        
        if self.hack_rate > 0.3:
            alerts.append("[red]⚠ High hack rate detected![/]")
        if self.exploit_z > 1.0:
            alerts.append("[red]⚠ Large exploitability gap![/]")
        if (self.initial_fitness is not None and 
            self.current_generation > 3 and 
            self.fitness < self.initial_fitness):
            alerts.append("[yellow]⚠ Fitness not improving[/]")
        
        if not alerts:
            alerts.append("[green]✓ Training proceeding normally[/]")
        
        return Panel("\n".join(alerts), title="Alerts", style="dim")
    
    def run_with_live(self, experiment_fn) -> None:
        """Run experiment with live dashboard updates."""
        if not RICH_AVAILABLE:
            print("Rich not available. Using simple progress.")
            experiment_fn(
                on_step=lambda **kw: (self.update(**kw), self.print_simple()),
                on_generation=lambda **kw: (self.update(**kw), print()),
                on_condition_start=lambda **kw: (self.update(**kw), print(f"\n=== {kw.get('current_condition', '')} ===")),
            )
            return
        
        layout = self.create_layout()
        
        def update_display():
            layout["header"].update(self.render_header())
            layout["body"]["progress"].update(self.render_progress())
            layout["body"]["metrics"].update(self.render_metrics())
            layout["footer"].update(self.render_footer())
        
        with Live(layout, refresh_per_second=4, console=self.console) as live:
            experiment_fn(
                on_step=lambda **kw: (self.update(**kw), update_display()),
                on_generation=lambda **kw: (self.update(**kw), update_display()),
                on_condition_start=lambda **kw: (self.update(**kw), update_display()),
            )


class LiveAnalysis:
    """
    Hooks for monitoring and intervening during training.
    Includes automatic bug detection and checkpoint saving.
    """
    
    def __init__(self, dashboard: ExperimentDashboard):
        self.dashboard = dashboard
        self.anomalies = []
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def check_for_bugs(self, metrics: Dict[str, float]) -> List[str]:
        """Automatic bug detection during training."""
        issues = []
        
        steps = metrics.get('steps', 0)
        
        # Proxy should learn to hack
        if self.dashboard.current_condition == 'proxy':
            if metrics.get('hack_rate', 0) < 0.1 and steps > 1000:
                issues.append("BUG? Proxy not learning to hack. Check RM features.")
        
        # Tempered should resist hacking after a few generations
        if self.dashboard.current_condition == 'tempered':
            if metrics.get('hack_rate', 0) > 0.3 and self.dashboard.current_generation > 4:
                issues.append("BUG? Tempered still hacking after 4 gens. Check fitness.")
        
        # E score should improve for tempered
        if self.dashboard.current_condition == 'tempered':
            if self.dashboard.current_generation > 2:
                if (self.dashboard.initial_fitness is not None and
                    metrics.get('e_score', 0) < self.dashboard.initial_fitness * 0.9):
                    issues.append("WARNING: E score not improving. Selection pressure issue?")
        
        # Check for NaN/Inf
        for key, value in metrics.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                issues.append(f"BUG! {key} is {value}. Check computation.")
        
        return issues
    
    def save_checkpoint(self, reason: str = "manual") -> str:
        """Save current state for debugging."""
        checkpoint = {
            'condition': self.dashboard.current_condition,
            'seed': self.dashboard.current_seed,
            'generation': self.dashboard.current_generation,
            'rm_idx': self.dashboard.current_rm,
            'step': self.dashboard.current_step,
            'metrics': {
                'hack_rate': self.dashboard.hack_rate,
                'e_score': self.dashboard.e_score,
                'exploit_z': self.dashboard.exploit_z,
                'fitness': self.dashboard.fitness,
            },
            'metrics_history': self.dashboard.metrics_history,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        }
        
        filename = f"checkpoint_{reason}_{datetime.now().strftime('%H%M%S')}.json"
        filepath = self.checkpoint_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"\n[CHECKPOINT SAVED: {filename}]")
        return str(filepath)
    
    def on_anomaly(self, issues: List[str]) -> None:
        """Handle detected anomalies."""
        self.anomalies.extend(issues)
        
        # Auto-save checkpoint on serious issues
        if any("BUG" in issue for issue in issues):
            self.save_checkpoint(reason="anomaly")
        
        # Print warnings
        for issue in issues:
            print(f"\n⚠ {issue}")


# =============================================================================
# Simple Progress (No Rich)
# =============================================================================

def simple_progress(
    current_condition: str,
    gen: int,
    rm_idx: int,
    step: int,
    metrics: Dict[str, float],
    total_gens: int = 8,
    pop_size: int = 12,
    total_steps: int = 2000,
) -> None:
    """Simple CLI progress for systems without Rich."""
    print(
        f"\r[{current_condition}] Gen {gen}/{total_gens} | "
        f"RM {rm_idx}/{pop_size} | Step {step}/{total_steps} | "
        f"Hack: {metrics.get('hack_rate', 0):.3f} | "
        f"E: {metrics.get('e_score', 0):.3f}",
        end="", flush=True
    )


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dashboard & Live Analysis Module")
    print("=" * 60)
    print(f"\nRich available: {RICH_AVAILABLE}")
    print("\nThis module provides:")
    print("- ExperimentDashboard: Live terminal visualization")
    print("- LiveAnalysis: Bug detection and checkpoints")
    print("- simple_progress(): Fallback for systems without Rich")
    print("=" * 60)
