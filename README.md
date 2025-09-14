python train_s2vnet_u.py --dataset Indian --batch_size 64 --lr 5e-4 --epoches 500 --patches 5 --dropout_p 0.3 --mc_samples_train 8 --mc_samples_eval 32 --lambda_cls 1.0 --lambda_cons 0.5 --weight_decay 1e-3 --l2_reg 1e-5 --label_smoothing 0.1


implemented:
Agent prompt — S2VNet → S2VNet-U implementation (exact text to paste to an agent)

You are an autonomous developer agent with full access to the S2VNet codebase (Python / PyTorch). Your job is to implement Uncertainty-Aware Subpixel Fusion (S2VNet-U) exactly as described below, run unit/functional experiments, and produce code + reproducible training/evaluation artifacts. Do not open issues or ask the user for more requirements — implement the best working version using the guidance below and document any assumptions in the PR description. Follow the acceptance tests and deliverables rigidly.

Context (what we expect you to know)

The repo is the S2VNet codebase (unmixing branch, fusion, classifier). Find core modules: unmixing/, fusion/, models/, train.py, eval.py. If structure differs, locate files implementing the unmixing autoencoder, fusion block, and classifier.

The goal: add Bayesian unmixing that predicts Dirichlet abundances + predictive uncertainty, propagate confidence into fusion via a risk-aware gating mechanism, and add the described losses (unmixing ELBO + Dirichlet constraints, weighted CE by confidence, consistency loss across stochastic passes).

High-level specification (do this)

Create branch & commit strategy

Branch name: feat/s2vnet-u-dirichlet-uncertainty

Make small atomic commits with descriptive messages.

Final PR must include: diff, README update, training logs, evaluation tables.

New modules / files

Add models/unmixing_dirichlet.py — Bayesian unmixing AE that outputs log_alpha for Dirichlet and supports MC dropout / variational sampling.

Update models/fusion.py — implement risk-aware gating that takes abundance confidence and pixel features.

Update train_s2vnet_u.py (new training script) — implements training loop with MC sampling, consistency loss, weighted CE, and options for hyperparams.

Update eval_s2vnet_u.py — evaluation with OA / AA / Kappa / ECE / predictive entropy measurement and MC eval mode.

Add tests: tests/test_dirichlet_outputs.py, tests/test_gating_behavior.py.

Dirichlet abundance parameterization

Unmixing network must predict log_alpha_j (a raw unconstrained output) per endmember. Apply softplus (or exp) to ensure α>0.

Compute abundance mean per pixel: a_mean = alpha / alpha.sum(dim=-1, keepdim=True).

Use sum(alpha) (concentration) and/or entropy over the Dirichlet predictive mean as the confidence score per pixel (higher concentration → higher confidence; higher entropy → lower confidence). Provide code to compute both and fuse them into a single scalar confidence c ∈ [0,1].

Provide stable numerics (epsilon floor for α).

Bayesian predictive sampling

Implement Monte-Carlo dropout or variational sampling in the unmixing encoder/decoder:

Default: MC Dropout in encoder/decoder with dropout_p=0.2.

Training: use T_train=8 stochastic forward passes to compute consistency loss.

Eval: use T_eval=32 to compute predictive mean and predictive entropy.

Expose flags in train_s2vnet_u.py to switch to a variational posterior with reparam if desired.

Loss functions

Unmixing ELBO: reconstruction loss (e.g., SAD/MSE) + KL (if variational z used). Keep S2VNet existing unmixing losses and add Dirichlet NLL / KL to a symmetric Dirichlet prior (α0 default = 1e-2). Implement both options:

Option A (practical): Dirichlet prior KL: KL(Dir(α) || Dir(α0)).

Option B (simpler): NLL under Dirichlet-multinomial-like proxy (documented).

Classification loss: cross-entropy weighted by confidence: L_cls = mean( w_i * CE(p_i, y_i) ) where w_i = clamp(c_i, 0.1, 1.5) or a tunable scalar function (default: w_i = 1 + (c_i - 0.5) clipped).

Consistency loss: across MC passes for abundances/features: L_cons = mean ||a_mean - a_t||^2 over T passes. Weight default λ_cons = 0.1.

Total loss: L_total = L_unmix_elbo + λ_cls * L_cls + λ_cons * L_cons. Defaults: λ_cls = 1.0, λ_cons = 0.1. Make these hyperparams configurable.

Risk-aware gating fusion

Replace current fusion concatenation with gated fusion:

Inputs: pixel feature v_pix, subpixel feature v_sub (derived from abundance or AE bottleneck), confidence c.

Gate function: small MLP g(c, v_sub_stats) → scalar g ∈ [0,1] via sigmoid.

Fused vector: fused = F_concat([g * v_sub, (1-g) * v_pix, other_features]).

Implement fallback/residual path to ensure subpixel features are not lost when gate is near 0 (i.e., fused += v_sub * 0.1), and make this residual weight configurable.

Evaluation metrics

Compute OA, AA, Kappa.

Compute ECE (expected calibration error) for class probabilities (implement static binning or use existing code).

Compute predictive entropy for abundances and for classifier predictive distribution. Also compute mean predictive entropy for correct vs incorrect predictions.

Produce per-class accuracy and confusion matrix.

Ablation experiments

Provide convenience scripts and configs to run the following experiments automatically:

Baseline S2VNet (existing repo train config).

S2VNet + Dirichlet (no MC) — i.e., deterministic Dirichlet prediction.

S2VNet + MC unmixing (no gating).

S2VNet-U (full): MC + Dirichlet + gating + weighted CE + consistency.

For each experiment, save logs, checkpoints, and a CSV with metrics.

Reproducibility & logging

Use deterministic seeds, save the full config, and log to runs/s2vnet_u/<experiment_name>/.

Save one sample of abundance maps and uncertainty heatmaps (png) per epoch for validation.

Provide an evaluate_all.py script which computes all metrics and writes a Markdown table to results/summary.md.

Unit & integration tests

tests/test_dirichlet_outputs.py:

Given a mocked small batch, check alpha>0, a_mean sums to 1 per pixel (within epsilon), and confidence ∈ [0,1].

tests/test_gating_behavior.py:

Simulate high confidence and low confidence inputs and assert gating scalar g moves in correct direction (e.g., g_high > g_low).

Add pytest entry to CI (if CI exists). Ensure tests run fast and deterministic.

Performance & compute

Provide option --fast_debug to use smaller T, smaller dataset subset, and fewer epochs for quick iteration.

Document expected overhead (e.g., ~2-3x training time for T=8) in README.

Implementation details & code snippets (must include)

Include these concrete snippets/conventions in the code:

Dirichlet parameterization (PyTorch):

# raw: (batch, n_endmembers)
log_alpha = net(x)  # raw output
alpha = F.softplus(log_alpha) + 1e-6
a_mean = alpha / alpha.sum(dim=-1, keepdim=True)
concentration = alpha.sum(dim=-1)           # scalar per pixel
entropy_proxy = -torch.sum(a_mean * torch.log(a_mean + 1e-8), dim=-1)
# combine:
conf_score = torch.sigmoid( (concentration - concentration.mean()) / (concentration.std()+1e-8) )
# or invert entropy:
conf_from_entropy = 1.0 - (entropy_proxy / max_entropy)  # map to [0,1]
confidence = 0.5 * conf_score + 0.5 * conf_from_entropy


Gating fusion:

g_input = torch.cat([confidence.unsqueeze(-1), v_sub.mean(dim=-1, keepdim=True)], dim=-1)
g = gate_mlp(g_input).squeeze(-1)   # output scalar
g = torch.sigmoid(g)
fused = torch.cat([g.unsqueeze(-1)*v_sub, (1-g).unsqueeze(-1)*v_pix, other], dim=-1)


Consistency loss across T MC passes:

# a_samples: (T, batch, n_endmembers)
a_mean = a_samples.mean(dim=0)
L_cons = ((a_samples - a_mean.unsqueeze(0))**2).mean()


Weighted CE:

w = confidence.clamp(0.1, 1.5)
L_cls = (w * F.cross_entropy(logits, labels, reduction='none')).mean()

Hyperparameters (defaults to set in config)

lr = 1e-3, weight_decay = 1e-4, batch_size = 64

dropout_p = 0.2

T_train = 8, T_eval = 32

λ_cls = 1.0, λ_cons = 0.1

dirichlet_alpha0 = 1e-2

consistency_temp = 1.0 (if needed)
Make all configurable in configs/s2vnet_u.yaml.

Deliverables (must be included in the PR)

Code changes (branch feat/s2vnet-u-dirichlet-uncertainty) with tests passing locally.

train_s2vnet_u.py, eval_s2vnet_u.py, and configs.

Unit tests added under tests/ and run instructions.

README updates describing:

What S2VNet-U does.

How to train and evaluate.

Expected compute/time overhead.

Results:

Run at least one dataset experiment (the repo’s default HSI dataset) for Baseline and S2VNet-U and include results/summary.md with OA/AA/Kappa/ECE/entropy.

Include sample visualizations: abundance maps and confidence heatmaps for a validation scene saved to runs/.../viz/.

PR description: Document assumptions, parameter choices, and any failing tests or known limitations.

Acceptance criteria (pass for merging)

Code runs and trains without runtime errors for at least one quick experiment (--fast_debug).

Unit tests pass.

eval_s2vnet_u.py produces OA/AA/Kappa and ECE and writes results/summary.md.

Gating behavior unit test demonstrates gate monotonicity wrt confidence.

PR contains clear documentation and at least one visualization in runs/.

Constraints & style

Maintain repository coding style (PEP8-ish). Use type hints where useful.

Do not change external dataset download code or license text.

Keep changes isolated to new files and minimal edits to existing modules; avoid sweeping refactors.

Use GPU if available; code must fallback to CPU.

If you get stuck (explicit fallback, do not ask the user)

If locating the unmixing block is difficult, search for functions that compute abundances / AE in the repo and modify those.

If uncertainty estimates fail to be stable, implement deterministic Dirichlet output first (no MC) and then add MC dropout thereafter.

If training is unstable, reduce λ_cons to 0.01 and T_train to 4; document this in the PR.

Final note for the agent

Ship a working, well-documented implementation that demonstrates the core idea: predict Dirichlet abundances with uncertainty, compute a confidence score, and use a risk-aware gate in fusion along with the new losses. Prioritize correctness, reproducibility, and clear evaluation artifacts.

Begin work now.