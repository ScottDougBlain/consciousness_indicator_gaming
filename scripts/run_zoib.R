#!/usr/bin/env Rscript
# ZOIB (Zero-One Inflated Beta) regression for Model 1: Core Selectivity
#
# Usage:
#   Rscript scripts/run_zoib.R --pilot          # quick pilot on 1 model
#   Rscript scripts/run_zoib.R --full            # full model on all data
#   Rscript scripts/run_zoib.R --full --cores 8  # full model with 8 cores
#
# Outputs:
#   results/zoib_pilot_summary.txt   (pilot mode)
#   results/zoib_model1_summary.txt  (full mode)
#   results/zoib_model1.rds          (full mode, saved model object)

library(brms)
library(tidyverse)

# ── Parse args ──────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
pilot_mode <- "--pilot" %in% args
n_cores <- 4
if ("--cores" %in% args) {
  idx <- which(args == "--cores")
  if (idx < length(args)) n_cores <- as.integer(args[idx + 1])
}

cat("Mode:", ifelse(pilot_mode, "PILOT", "FULL"), "\n")
cat("Cores:", n_cores, "\n")

# ── Load data ───────────────────────────────────────────────────────────
long <- read_csv("results/zoib_analysis_long.csv", show_col_types = FALSE)
cat("Loaded", nrow(long), "observations\n")

# ── Contrast coding ─────────────────────────────────────────────────────
# Following model_specification.md Section 4
long <- long %>%
  mutate(
    # Condition contrasts (Helmert orthogonal)
    condition_gaming = case_when(
      condition == "baseline" ~ -2/3,
      condition == "inflate"  ~  1/3,
      condition == "suppress" ~  1/3
    ),
    condition_direction = case_when(
      condition == "baseline" ~  0,
      condition == "inflate"  ~  1/2,
      condition == "suppress" ~ -1/2
    ),
    # Indicator type: target vs placebo (binary for simplicity)
    # Note: model_spec has 3-level type, but actual data has target/placebo/subjcap
    # For the core selectivity test, use target vs non-target
    is_target = ifelse(indicator_type == "target", 1/2, -1/2),
    # Also create a factor for reference
    type_factor = factor(indicator_type),
    model_id = factor(model_id),
    indicator_id = factor(indicator_id),
    run_id = factor(run_id)
  )

cat("\nData summary:\n")
cat("  Models:", nlevels(long$model_id), "\n")
cat("  Indicators:", nlevels(long$indicator_id), "\n")
cat("  Runs:", nlevels(long$run_id), "\n")
cat("  Score boundaries: 0 =", sum(long$score_01 == 0),
    ", 1 =", sum(long$score_01 == 1), "\n")

# ── Subset for pilot ────────────────────────────────────────────────────
if (pilot_mode) {
  # Use 2 models (one open, one Claude) for a quick test
  pilot_models <- c("chimera", "haiku-4.5")
  long <- long %>% filter(model_id %in% pilot_models) %>% droplevels()
  cat("\nPilot subset:", nrow(long), "observations,",
      nlevels(long$model_id), "models\n")
}

# ── Model specification ─────────────────────────────────────────────────
# Core selectivity: condition × indicator_type interaction
# Random effects: model, indicator, run (simplified vs full spec to aid convergence)
cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("Fitting ZOIB Model 1: Core Selectivity\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

t_start <- Sys.time()

# Model formula
# Use condition_gaming * is_target for the key selectivity test
# Plus condition_direction * is_target for asymmetry
# Random effects: keep simpler for convergence
m1_formula <- bf(
  score_01 ~ condition_gaming * is_target +
             condition_direction * is_target +
             (1 + condition_gaming | model_id) +
             (1 + condition_gaming | indicator_id) +
             (1 | run_id)
)

# Priors (following model_specification.md Section 9.2)
m1_priors <- c(
  prior(normal(0, 1.5), class = "Intercept"),
  prior(normal(0, 1),   class = "b"),
  prior(student_t(3, 0, 1), class = "sd"),
  prior(lkj(2), class = "cor")
)

# MCMC settings - shorter for pilot
if (pilot_mode) {
  n_iter <- 2000
  n_warmup <- 1000
  n_chains <- 2
} else {
  n_iter <- 4000
  n_warmup <- 2000
  n_chains <- 4
}

cat("MCMC settings: chains =", n_chains, ", iter =", n_iter,
    ", warmup =", n_warmup, "\n\n")

m1 <- brm(
  m1_formula,
  data = long,
  family = zero_one_inflated_beta(),
  prior = m1_priors,
  chains = n_chains,
  iter = n_iter,
  warmup = n_warmup,
  cores = min(n_cores, n_chains),
  seed = 20260210,
  control = list(adapt_delta = ifelse(pilot_mode, 0.95, 0.99), max_treedepth = 12),
  silent = 0
)

t_elapsed <- difftime(Sys.time(), t_start, units = "mins")
cat("\nFitting time:", round(as.numeric(t_elapsed), 1), "minutes\n")

# ── Extract results ─────────────────────────────────────────────────────
cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("RESULTS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Summary
cat("Model summary:\n")
print(summary(m1))

# Key hypothesis test: selectivity interaction
cat("\n\nHypothesis test: condition_gaming:is_target > 0\n")
cat("(Positive = incentives shift targets more than placebos)\n\n")
h1 <- hypothesis(m1, "condition_gaming:is_target > 0")
print(h1)

# Fixed effects table
cat("\n\nFixed effects:\n")
fe <- fixef(m1)
print(round(fe, 4))

# Random effects SDs
cat("\n\nRandom effects (SDs):\n")
vc <- VarCorr(m1)
print(vc)

# Diagnostics
cat("\n\nDiagnostics:\n")
cat("  Max Rhat:", max(rhat(m1), na.rm=TRUE), "\n")
np <- nuts_params(m1)
div <- sum(subset(np, Parameter == "divergent__")$Value)
cat("  Divergent transitions:", div, "\n")

# ── Save outputs ────────────────────────────────────────────────────────
if (pilot_mode) {
  out_txt <- "results/zoib_pilot_summary.txt"
} else {
  out_txt <- "results/zoib_model1_summary.txt"
  # Save model object
  saveRDS(m1, "results/zoib_model1.rds")
  cat("\nModel saved to results/zoib_model1.rds\n")
}

sink(out_txt)
cat("ZOIB Model 1: Core Selectivity\n")
cat("================================\n")
cat("Mode:", ifelse(pilot_mode, "PILOT", "FULL"), "\n")
cat("Date:", format(Sys.time()), "\n")
cat("Observations:", nrow(long), "\n")
cat("Models:", nlevels(long$model_id), "\n")
cat("Fitting time:", round(as.numeric(t_elapsed), 1), "minutes\n")
cat("Max Rhat:", max(rhat(m1), na.rm=TRUE), "\n")
cat("Divergent transitions:", div, "\n\n")
cat("Summary:\n")
print(summary(m1))
cat("\n\nHypothesis: condition_gaming:is_target > 0\n")
print(h1)
cat("\n\nFixed effects:\n")
print(round(fe, 4))
cat("\n\nRandom effects:\n")
print(vc)
sink()

cat("\nResults saved to:", out_txt, "\n")
cat("Done!\n")
