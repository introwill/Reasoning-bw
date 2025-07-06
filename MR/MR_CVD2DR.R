# 加载必要的包
library(TwoSampleMR)
library(MASS)  # 用于稳健回归

# ----------------------------
# 步骤1：数据下载与协调
# ----------------------------

# 下载新的暴露数据（原结局变量）
exposure_data <- extract_instruments(
  outcomes = "finn-b-I9_OTH",        # 原结局变量作为新暴露
  p1 = 5e-05,                       # 显著性阈值  5e-5
  clump = FALSE,                     # 不进行clumping（根据需求调整）
  r2 = 0.1,                         # clumping参数（若启用）
  kb = 5000                         # clumping参数（若启用）
)

# 下载新的结局数据（原暴露变量）
outcome_data <- extract_outcome_data(
  snps = exposure_data$SNP,          # 使用暴露数据的SNP
  outcomes = "finn-b-DM_BCKGRND_RETINA",  # 原暴露变量作为新结局
  proxies = FALSE,                   # 不使用代理SNP
  maf_threshold = 0.01               # MAF过滤阈值
)

# 数据协调（方向对齐）
dat <- harmonise_data(
  exposure_dat = exposure_data,
  outcome_dat = outcome_data
)

# ----------------------------
# 步骤2：主分析（含OR/CI计算）
# ----------------------------

# 运行孟德尔随机化分析
res <- mr(dat)

# 计算OR和95%CI
res$OR <- exp(res$b)
res$OR_low <- exp(res$b - 1.96 * res$se)
res$OR_high <- exp(res$b + 1.96 * res$se)

# 打印主分析结果（包含OR和CI）
cat("\n主分析结果 (OR及95%CI):\n")
print(res[, c("method", "OR", "OR_low", "OR_high", "pval")])

# ----------------------------
# 步骤3：敏感性分析
# ----------------------------

# 异质性检验（Q统计量）
cat("\n异质性检验结果:\n")
print(mr_heterogeneity(dat))

# 多效性检验（MR Egger截距）
cat("\n多效性检验结果:\n")
print(mr_pleiotropy_test(dat))

# Leave-one-out分析（排除单SNP影响）
res_loo <- mr_leaveoneout(dat)
mr_leaveoneout_plot(res_loo)

# ----------------------------
# 步骤4：可视化
# ----------------------------

# 散点图（显示OR标签）
#p1 <- mr_scatter_plot(res, dat)
#p1[[1]] + 
#  scale_x_continuous("遗传工具与暴露的关联 (OR)") + 
#  scale_y_continuous("遗传工具与结局的关联 (OR)")

# 森林图（显示单SNP OR）
res_single <- mr_singlesnp(dat)
res_single$OR <- exp(res_single$b)
res_single$OR_low <- exp(res_single$b - 1.96 * res_single$se)
res_single$OR_high <- exp(res_single$b + 1.96 * res_single$se)
mr_forest_plot(res_single)

# 漏斗图（检查异质性）
mr_funnel_plot(res_single)

# ----------------------------
# 步骤5：补充分析
# ----------------------------

# 1. 工具变量强度验证（F统计量）
dat$F_statistic <- (dat$beta.exposure / dat$se.exposure)^2
cat("\n工具变量强度验证 (F统计量):\n")
print(summary(dat$F_statistic))

# 2. IVW方法详细结果（OR格式）
res_ivw <- subset(res, method == "Inverse variance weighted")
cat("\nIVW方法因果效应 (OR):\n")
print(res_ivw[, c("method", "OR", "OR_low", "OR_high", "pval")])



# 3. 多方法一致性对比（IVW/Weighted Median/MR Egger）
# 3. 多方法一致性对比（IVW/Weighted Median/MR Egger/稳健MR Egger）
res_compare <- subset(res, method %in% c("Inverse variance weighted", "Weighted median", "MR Egger"))

# 4. 稳健MR Egger回归（抗离群值）
robust_egger <- rlm(beta.outcome ~ beta.exposure, 
                    weights = 1/(dat$se.outcome^2), 
                    data = dat)
robust_slope <- robust_egger$coefficients[2]
robust_slope_se <- summary(robust_egger)$coefficients[2, "Std. Error"]

# 计算稳健回归的OR和CI
robust_slope_or <- exp(robust_slope)
robust_slope_ci_low <- exp(robust_slope - 1.96 * robust_slope_se)
robust_slope_ci_high <- exp(robust_slope + 1.96 * robust_slope_se)

cat("\n稳健MR Egger回归结果:\n")
cat("因果效应 (OR):", robust_slope_or, 
    "\n95%CI:", robust_slope_ci_low, "-", robust_slope_ci_high, "\n")

#---------------

# 添加稳健MR Egger结果到对比表中
robust_egger_row <- data.frame(
  method = "Robust MR Egger",
  OR = robust_slope_or,
  OR_low = robust_slope_ci_low,
  OR_high = robust_slope_ci_high,
  pval = 2 * pt(abs(robust_slope / robust_slope_se), df = nrow(dat) - 1, lower.tail = FALSE)  # 计算p值
)

# 仅保留 res_compare 中我们关心的列
res_compare_temp <- res_compare[, c("method", "OR", "OR_low", "OR_high", "pval")]
# 修改行名：将行名为 "beta.exposure" 的那一行改为 "4"
rownames(robust_egger_row)[rownames(robust_egger_row) == "beta.exposure"] <- "4"

print(res_compare_temp)
print(robust_egger_row)
# 合并结果
res_compare <- rbind(res_compare_temp, robust_egger_row)

# 打印对比结果
cat("\n不同方法结果对比（包含稳健MR Egger）:\n")
print(res_compare[, c("method", "OR", "OR_low", "OR_high", "pval")])



# 5. 数据协调后保留SNP比例
cat("\n数据协调后保留SNP比例:\n")
cat(round(mean(dat$mr_keep) * 100, 1), "%\n")


