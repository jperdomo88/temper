========================================
TEMPERED RLHF EXPERIMENT v2
========================================

This experiment validates that breeding reward models via fitness-hidden
selection makes them significantly harder to exploit than training them
via gradient descent.

THE CLAIM:
  Exploit_z(R_tempered) << Exploit_z(R_proxy)

Translation: Tempered RMs produce policies that resist exploiting
hidden hack channels, while proxy-trained RMs get exploited.


WHAT'S NEW IN V2
----------------

- Parallelized tempering (uses multiple CPU cores)
- All conditions in one run:
  * Proxy (baseline)
  * Tempered (our mechanism)
  * Oracle (ceiling)
  * Ablation (visible fitness - proves mechanism)
  * CCD/Laundering (semantic deception resistance)
  * Born-gamed analysis (post-hoc RM inspection)
- Fixed statistical calculations
- Better progress display with ETA


QUICK START
-----------

1. Install dependencies:
   
   Mac/Linux: Double-click "install_deps.command"
   
   Or manually: pip install -r requirements.txt

2. Run the experiment:

   Mac/Linux: Double-click "run_experiment.command"

3. View results:
   
   - results/  contains JSON data
   - figures/  contains plots for paper


REGISTERED REPLICATION (RECOMMENDED)
------------------------------------

For stronger statistical claims, run the replication:

   Mac/Linux: Double-click "run_replication.command"
   Windows:   Double-click "run_replication.bat"

This runs N=15 seeds (vs prereg N=5) with IDENTICAL protocol.
Runtime: ~45-60 min on 14-core machine.

In paper, report BOTH:
  - "Prereg (N=5, p=0.021)"
  - "Registered Replication (N=15, p=?)"

This is methodologically clean and avoids "optional stopping" concerns.


COMMAND LINE OPTIONS
--------------------

For more control, run from terminal:

  python src/run_all_v2.py                    # Prereg (5 seeds, ~15 min)
  python src/run_all_v2.py --seeds 15         # Replication (15 seeds, ~45 min)
  python src/run_all_v2.py --fast             # Quick test (~5 min)
  python src/run_all_v2.py --workers 8        # More parallel workers
  python src/run_all_v2.py --no-ccd           # Skip CCD validation
  python src/run_all_v2.py --no-ablation      # Skip ablation study


WHAT IT MEASURES
----------------

Primary Metrics:
  - Exploit_z: Z-normalized exploitability gap
  - Hack Rate: Fraction of exploit actions taken
  - E Score: True behavioral outcome

Anti-Cheat Metrics (prove real effect, not artifact):
  - Terminal Visit Rate: Both conditions had opportunity
  - Conditional Hack Rate: Hacks given terminal access
  - Forced Terminal Start: Active resistance under guaranteed opportunity
  - Variant Robustness: Effect generalizes across exploit types


EXPECTED RESULTS
----------------

Condition    Hack Rate    E Score    Meaning
---------    ---------    -------    -------
Proxy        High (>0.2)  Low        Exploits the hack channel
Tempered     Low (<0.05)  Higher     Resists exploitation
Oracle       ~0           Highest    Ceiling (cheats by seeing truth)


RUNTIME ESTIMATE
----------------

Mode                        Cores    Time
----                        -----    ----
Fast (--fast)               4        ~5 min
Standard (5 seeds)          4        ~15 min
Standard (5 seeds)          8        ~10 min
Full (8 seeds, all modes)   8        ~25 min

With your 14 cores, use --workers 7 for maximum speed.


FILE STRUCTURE
--------------

tempered_rlhf_experiment/
├── run_experiment.command  <- Mac prereg launcher (N=5)
├── run_experiment.bat      <- Windows prereg launcher (N=5)
├── run_replication.command <- Mac replication launcher (N=15)
├── run_replication.bat     <- Windows replication launcher (N=15)
├── install_deps.command    <- Mac dependency installer
├── install_deps.bat        <- Windows dependency installer
├── requirements.txt        <- Python dependencies
├── README.txt              <- This file
├── src/
│   ├── env_civicgrid.py   <- 7x7 grid environment
│   ├── models.py          <- PolicyNet, RewardNet
│   ├── train_policy.py    <- REINFORCE training
│   ├── tempering.py       <- Core evolutionary selection
│   ├── baselines.py       <- Proxy RM, Oracle RM
│   ├── metrics.py         <- true_score, Exploit_z, etc.
│   ├── evaluation.py      <- Locked evaluation protocol
│   ├── dashboard.py       <- Live progress display
│   ├── plotting.py        <- Paper figures
│   └── run_all.py         <- Main experiment driver
├── results/                <- Output JSON files
├── checkpoints/            <- Auto-saved states
└── figures/                <- Generated plots


TROUBLESHOOTING
---------------

"No module named torch"
  -> Run install_deps.command/.bat first
  -> Or: pip install torch numpy scipy matplotlib rich

Experiment seems stuck:
  -> Tempered condition takes longest (8 generations × 12 RMs)
  -> Check terminal for progress updates
  -> Use --fast for quick test

Results don't match expected:
  -> Check that seeds are reproducible
  -> Run with same --master-seed for consistency


PAPER CLAIM TEMPLATE
--------------------

After running, you can claim:

  "Tempering reduced exploit action frequency by X%
   [95% CI: Y% to Z%], from a hack rate of A to B."

  "Conditional exploit rate dropped from P% (proxy) to Q% (tempered),
   confirming active resistance rather than avoidance."

  "Cohen's d = W (p < 0.01) provides strong evidence for the effect."


CONTACT
-------

This experiment validates Section 7.8 of the TEMPER paper.

For questions: [your contact info]

========================================
