# Load necessary packages
library(TwoSampleMR)
library(MASS)  # For robust regression

# ----------------------------
# Step 1: Data download and harmonization
# ----------------------------

# Download new exposure data (original outcome variable)
exposure_data <- extract_instruments(
  outcomes = "finn-b-I9_OTH",        # Original outcome variable as new exposure
  p1 = 5e-05,                       # Significance threshold 5e-5
  clump = FALSE,                     # No clumping (adjust as needed)
  r2 = 0.1,                         # Clumping parameter (if enabled)
  kb = 5000                         # Clumping parameter (if enabled)
)

# Download new outcome data (original exposure variable)
outcome_data <- extract_outcome_data(
  snps = exposure_data$SNP,          # Use SNPs from exposure data
  outcomes = "finn-b-DM_BCKGRND_RETINA",  # Original exposure variable as new outcome
  proxies = FALSE,                   # Do not use proxy SNPs
  maf_threshold = 0.01               # MAF filtering threshold
)

# Data harmonization (direction alignment)
dat <- harmonise_data(
  exposure_dat = exposure_data,
  outcome_dat = outcome_data
)

# ----------------------------
# Step 2: Main analysis (including OR/CI calculation)
# ----------------------------

# Run Mendelian randomization analysis
res <- mr(dat)

# Calculate OR and 95%CI
res$OR <- exp(res$b)
res$OR_low <- exp(res$b - 1.96 * res$se)
res$OR_high <- exp(res$b + 1.96 * res$se)

# Print main analysis results (including OR and CI)
cat("\nMain Analysis Results (OR and 95%CI):\n")
print(res[, c("method", "OR", "OR_low", "OR_high", "pval")])

# ----------------------------
# Step 3: Sensitivity analysis
# ----------------------------

# Heterogeneity test (Q statistic)
cat("\nHeterogeneity Test Results:\n")
print(mr_heterogeneity(dat))

# Pleiotropy test (MR Egger intercept)
cat("\nPleiotropy Test Results:\n")
print(mr_pleiotropy_test(dat))

# Leave-one-out analysis (exclude single SNP effects)
res_loo <- mr_leaveoneout(dat)
mr_leaveoneout_plot(res_loo)

# ----------------------------
# Step 4: Visualization
# ----------------------------

# Scatter plot (display OR labels)
#p1 <- mr_scatter_plot(res, dat)
#p1[[1]] + 
#  scale_x_continuous("Association between genetic instruments and exposure (OR)") + 
#  scale_y_continuous("Association between genetic instruments and outcome (OR)")

# Forest plot (display single SNP OR)
res_single <- mr_singlesnp(dat)
res_single$OR <- exp(res_single$b)
res_single$OR_low <- exp(res_single$b - 1.96 * res_single$se)
res_single$OR_high <- exp(res_single$b + 1.96 * res_single$se)
mr_forest_plot(res_single)

# Funnel plot (check heterogeneity)
mr_funnel_plot(res_single)

# ----------------------------
# Step 5: Supplementary analysis
# ----------------------------

# 1. Instrumental variable strength validation (F statistic)
dat$F_statistic <- (dat$beta.exposure / dat$se.exposure)^2
cat("\nInstrumental Variable Strength Validation (F statistic):\n")
print(summary(dat$F_statistic))

# 2. IVW method detailed results (OR format)
res_ivw <- subset(res, method == "Inverse variance weighted")
cat("\nIVW Method Causal Effect (OR):\n")
print(res_ivw[, c("method", "OR", "OR_low", "OR_high", "pval")])



# 3. Multi-method consistency comparison (IVW/Weighted Median/MR Egger/Robust MR Egger)
res_compare <- subset(res, method %in% c("Inverse variance weighted", "Weighted median", "MR Egger"))

# 4. Robust MR Egger regression (outlier-resistant)
robust_egger <- rlm(beta.outcome ~ beta.exposure, 
                    weights = 1/(dat$se.outcome^2), 
                    data = dat)
robust_slope <- robust_egger$coefficients[2]
robust_slope_se <- summary(robust_egger)$coefficients[2, "Std. Error"]

# Calculate OR and CI for robust regression
robust_slope_or <- exp(robust_slope)
robust_slope_ci_low <- exp(robust_slope - 1.96 * robust_slope_se)
robust_slope_ci_high <- exp(robust_slope + 1.96 * robust_slope_se)

cat("\nRobust MR Egger Regression Results:\n")
cat("Causal Effect (OR):", robust_slope_or, 
    "\n95%CI:", robust_slope_ci_low, "-", robust_slope_ci_high, "\n")

#---------------

# Add robust MR Egger results to comparison table
robust_egger_row <- data.frame(
  method = "Robust MR Egger",
  OR = robust_slope_or,
  OR_low = robust_slope_ci_low,
  OR_high = robust_slope_ci_high,
  pval = 2 * pt(abs(robust_slope / robust_slope_se), df = nrow(dat) - 1, lower.tail = FALSE)  # Calculate p-value
)

# Keep only the columns we care about in res_compare
res_compare_temp <- res_compare[, c("method", "OR", "OR_low", "OR_high", "pval")]
# Modify row names: change the row name "beta.exposure" to "4"
rownames(robust_egger_row)[rownames(robust_egger_row) == "beta.exposure"] <- "4"

print(res_compare_temp)
print(robust_egger_row)
# Merge results
res_compare <- rbind(res_compare_temp, robust_egger_row)

# Print comparison results
cat("\nComparison of Different Methods (including Robust MR Egger):\n")
print(res_compare[, c("method", "OR", "OR_low", "OR_high", "pval")])



# 5. Proportion of SNPs retained after data harmonization
cat("\nProportion of SNPs retained after data harmonization:\n")
cat(round(mean(dat$mr_keep) * 100, 1), "%\n")


