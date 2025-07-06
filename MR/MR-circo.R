# ==============================
# Load these packages at the beginning of your script
# ==============================
library(TwoSampleMR)
library(GenomicRanges)
library(dplyr)
library(circlize)
library(ComplexHeatmap)  # For colorRamp2 function
library(viridis)      # 提供 cividis、magma 等色带
library(RColorBrewer) # 提供经典调色板


# —— 1. 热图色带 ——  
# 用 viridis::cividis，视觉平滑且对色盲友好
max_den <- max(dens_exp$density, na.rm = TRUE)
col_fun_heat <- colorRamp2(
  breaks = c(0, max_den/4, max_den/2, 3*max_den/4, max_den),
  colors = viridis(5, option = "cividis")
)

# —— 2. 染色体背景色 ——  
# 全部用浅灰色，让热图和散点更突出
chr_list <- paste0("chr", c(1:22, "X", "Y"))
chr_cols <- setNames(
  rep("#EEEEEE", length(chr_list)),
  chr_list
)

# —— 3. 曝露/结局组散点色 ——  
# 选用 Nature 期刊常用的高对比蓝—橙配色
group_cols <- c(
  exposure = "#0072B2",  # 深蓝
  outcome  = "#D55E00"   # 橙红
)

# —— 4. 阈值线颜色 ——  
# 画成深灰，低调不抢散点风头
th_col <- "#333333"

# New: Load BSgenome package to get chromosome lengths
if (!requireNamespace("BSgenome.Hsapiens.UCSC.hg19", quietly = TRUE)) {
  BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
}
library(BSgenome.Hsapiens.UCSC.hg19)


# ------------------------------
# 1. Download GWAS and coordinate
# ------------------------------
# Exposure: finn-b-I9_OTH
exposure_dat <- extract_instruments(
  outcomes = "finn-b-I9_OTH",
  p1       = 5e-5,    # Only keep genome-wide significant
  clump    = FALSE
)
# Outcome: finn-b-DM_BCKGRND_RETINA
outcome_dat  <- extract_outcome_data(
  snps    = exposure_dat$SNP,
  outcomes= "finn-b-DM_BCKGRND_RETINA"
)
dat <- harmonise_data(exposure_dat, outcome_dat)

# We need CHR, POS, P columns
# Extract exposure information
gwas_exp <- dat %>%
  select(
    SNP,
    CHR = chr.exposure,
    POS = pos.exposure,
    P   = pval.exposure
  )

# Extract outcome information
gwas_out <- dat %>%
  select(
    SNP,
    CHR = chr,
    POS = pos,
    P   = pval.outcome
  )



# ------------------------------
# 2. Construct 1Mb sliding window (improved version)
# ------------------------------
# Get chromosome lengths directly from BSgenome.Hsapiens.UCSC.hg19
genome <- BSgenome.Hsapiens.UCSC.hg19
seqlens <- seqlengths(genome)

# Only keep main chromosomes
main_chrs <- paste0("chr", c(1:22, "X", "Y"))
seqlens_filtered <- seqlens[names(seqlens) %in% main_chrs]

chrom_sizes <- data.frame(
  chr    = names(seqlens_filtered),           # Chromosome name
  length = as.numeric(seqlens_filtered),      # Length (bp)
  row.names = NULL
)

# Create windows using filtered chromosome lengths
tiles <- tileGenome(
  seqlengths = seqlens_filtered,
  tilewidth  = 1e6,
  cut.last.tile.in.chrom = TRUE
)

# ------------------------------
# 3. Filter significant SNPs for scatter plot
# ------------------------------
# Helper function: Calculate SNP density per window (modified part)
calc_density <- function(gwas_df){
  # Ensure POS is numeric
  pos_num <- as.numeric(gwas_df$POS)
  if (any(is.na(pos_num))) {
    stop("POS column contains values that cannot be converted to numeric, please check gwas_df$POS")
  }
  
  # Ensure CHR is character type
  chr_str <- paste0("chr", gwas_df$CHR)
  
  # Only keep data for main chromosomes
  valid_idx <- chr_str %in% main_chrs
  if(sum(valid_idx) == 0) {
    warning("No data found for main chromosomes")
    return(data.frame())
  }
  
  # Construct GRanges
  gr <- GRanges(
    seqnames = chr_str[valid_idx],
    ranges   = IRanges(start = pos_num[valid_idx], width = 1L)
  )
  
  # Count overlaps for each tile
  cnt <- countOverlaps(tiles, gr)
  
  # Return data.frame
  data.frame(
    chr     = as.character(seqnames(tiles)),
    start   = start(tiles),
    end     = end(tiles),
    density = cnt,
    stringsAsFactors = FALSE
  )
}

# Then call
dens_exp <- calc_density(gwas_exp)
dens_out <- calc_density(gwas_out)

# Helper function: Maximum –log10(P) per window, only use main chromosomes
calc_maxlp <- function(gwas_df){
  # Create data frame and explicitly specify column names
  df2 <- gwas_df %>% 
    mutate(chr = paste0("chr", CHR), 
           lp = -log10(P),
           POS = as.numeric(POS)) %>%
    filter(chr %in% main_chrs)  # Only keep main chromosomes
  
  # If no data, return empty data frame
  if(nrow(df2) == 0) {
    return(data.frame(chr=character(), start=numeric(), end=numeric(), value=numeric()))
  }
  
  # Directly build GRanges object
  gr <- GRanges(
    seqnames = df2$chr,
    ranges = IRanges(start = df2$POS, width = 1L)
  )
  
  # Add lp as metadata column
  mcols(gr)$lp <- df2$lp
  
  # Set chromosome levels consistent with filtered chromosomes
  seqlevels(gr) <- names(seqlens_filtered)
  seqlengths(gr) <- seqlens_filtered
  
  # Generate coverage RleList
  cov_rle <- coverage(gr, weight = "lp")
  
  # Aggregate by window
  m <- suppressWarnings(binnedAverage(tiles, cov_rle, "value"))
  
  data.frame(chr = seqnames(m),
             start = start(m),
             end = end(m),
             value = ifelse(is.na(m$value), 0, m$value))
}

# Then call
hist_exp <- calc_maxlp(gwas_exp)
hist_out <- calc_maxlp(gwas_out)

# Modify sig_scatter function, use more relaxed threshold and only keep main chromosomes
sig_scatter <- function(gwas_df, label){
  gwas_df %>% 
    filter(P < 5e-5) %>%  # Use more relaxed threshold
    transmute(chr = paste0("chr", CHR),
              pos = as.numeric(POS),
              lp  = -log10(P),
              group = label) %>%
    filter(chr %in% main_chrs)  # Only keep main chromosomes
}

scat_exp <- sig_scatter(gwas_exp, "exposure")
scat_out <- sig_scatter(gwas_out, "outcome")
scat_all <- bind_rows(scat_exp, scat_out)






# ------------------------------
# 4. 用 circlize 画出：外圈热力 + 多圈彩色散点曼哈顿
# ------------------------------

png("A_group_circos_beautiful2.png",
    width = 3000, height = 3000, res = 600)

library(RColorBrewer)
# 为每条染色体准备不同颜色(共24)
chr_list <- paste0("chr", c(1:22, "X", "Y"))
chr_cols <- setNames(
  brewer.pal(12, "Set3")[rep(1:12, each=2)][1:24],
  chr_list
)

# heatmap 调色函数 (0–95%分位数映射到白→红)
max_den <- quantile(dens_exp$density, 0.95)
col_fun_heat <- colorRamp2(c(0, max_den), c("white", "red"))

circos.clear()
circos.par(
  start.degree = 90,
  gap.degree   = 1,
  cell.padding = c(0,0,0,0),
  track.margin = c(0.02,0.02)
)

# 初始化基于染色体长度
circos.initialize(
  factors = chrom_sizes$chr,
  xlim    = matrix(c(rep(0, nrow(chrom_sizes)),
                     chrom_sizes$length),
                   ncol=2)
)

# === 最外圈：密度热力图 ===
# 1. 重新设置热图色带：用全部最大密度
max_den <- max(dens_exp$density, na.rm = TRUE)
col_fun_heat <- colorRamp2(
  breaks = c(0, max_den/4, max_den/2, 3*max_den/4, max_den),
  colors = c("white", "lightblue", "yellow", "orange", "red")
)

# === 最外圈：密度热力图 ===
circos.trackPlotRegion(
  track.height = 0.08,
  bg.border    = NA,
  ylim         = c(0, 1),
  panel.fun    = function(x, y) {
    chr <- CELL_META$sector.index
    df  <- dens_exp[dens_exp$chr == chr, ]
    if (nrow(df) > 0) {
      for (i in seq_len(nrow(df))) {
        circos.rect(
          xleft   = df$start[i],
          ybottom = 0,
          xright  = df$end[i],
          ytop    = 1,
          col     = col_fun_heat(df$density[i]),
          border  = NA
        )
      }
    }
  }
)

# === 第二圈：曝光组曼哈顿散点 ===
circos.trackPlotRegion(
  track.height = 0.12,
  bg.border    = NA,
  ylim         = c(0, max_lp_exp * 1.1),
  panel.fun    = function(x, y) {
    chr <- CELL_META$sector.index
    df  <- gwas_exp %>%
      filter(paste0("chr", CHR) == chr) %>%
      mutate(lp = -log10(P))
    if (nrow(df) > 0) {
      circos.points(
        df$POS,
        df$lp,
        col = chr_cols[chr],
        pch = 16,
        cex = 0.5
      )
    }
  }
)

# === 第三圈：结局组曼哈顿散点 ===
circos.trackPlotRegion(
  track.height = 0.12,
  bg.border    = NA,
  ylim         = c(0, max_lp_out * 1.1),
  panel.fun    = function(x, y) {
    chr <- CELL_META$sector.index
    df <- gwas_out %>%
      filter(paste0("chr", CHR) == chr) %>%
      mutate(
        lp  = -log10(as.numeric(P)),
        POS = as.numeric(POS)
      )
    if (nrow(df) > 0) {
      circos.points(
        df$POS,
        df$lp,
        col = chr_cols[chr],
        pch = 17,
        cex = 0.5
      )
    }
  }
)

# === 第四层：合并组散点 + 阈值线 ===
circos.trackPlotRegion(
  track.height = 0.15,  # 增加轨道高度
  bg.border    = NA,
  ylim         = c(0, max_lp_all * 1.1),
  panel.fun    = function(x, y) {
    chr <- CELL_META$sector.index
    df  <- scat_all %>%
      filter(chr == chr)
    if (nrow(df) > 0) {
      # 散点，添加透明度
      circos.points(
        df$pos,
        df$lp,
        col = adjustcolor(group_cols[df$group], alpha.f = 0.6),
        pch = ifelse(df$group == "exposure", 16, 17),
        cex = 0.4  # 减小点的大小
      )
      # 阈值线
      xlim <- CELL_META$xlim
      circos.lines(
        x   = xlim,
        y   = rep(th, 2),
        col = th_col,
        lty = 2,
        lwd = 1.5  # 增加阈值线宽度
      )
      # 标注同时显著的 SNPs
      df_both <- df %>% filter(lp > th)
      if (nrow(df_both) > 0) {
        circos.points(
          df_both$pos,
          df_both$lp,
          col = adjustcolor("purple", alpha.f = 0.8),
          pch = 3,
          cex = 0.7
        )
      }
    }
  }
)

# === 染色体标签和基因注释 ===
circos.trackPlotRegion(
  track.index = 1,
  bg.border    = NA,
  panel.fun    = function(x, y) {
    circos.text(
      CELL_META$xcenter,
      CELL_META$ylim[2] + mm_y(2),
      gsub("chr", "", CELL_META$sector.index),
      facing       = "clockwise",
      niceFacing   = TRUE,
      cex          = 0.7  # 调整标签字体大小
    )
    # 添加关键基因注释
    gene_pos <- data.frame(
      gene = c("GeneA", "GeneB"),
      pos = c(12345678, 87654321),
      chr = c("chr1", "chr2")
    )
    if (CELL_META$sector.index %in% gene_pos$chr) {
      gene_sub <- gene_pos %>% filter(chr == CELL_META$sector.index)
      for (i in seq_len(nrow(gene_sub))) {
        circos.text(
          gene_sub$pos[i],
          CELL_META$ylim[2] + mm_y(4),
          gene_sub$gene[i],
          col = "blue",
          cex = 0.5  # 调整基因注释字体大小
        )
      }
    }
  }
)

# === 图例和注释 ===
legend(
  "bottom",
  legend = c("Exposure SNP", "Outcome SNP", "Significant in Both", "Threshold"),
  pch    = c(16, 17, 3, NA),
  col    = c(group_cols["exposure"], group_cols["outcome"], "purple", th_col),
  lty    = c(NA, NA, NA, 2),
  pt.cex = 1,
  bty    = "n"
)
mtext(
  "Key genes: GeneA (chr1), GeneB (chr2)",
  side = 1,
  line = 2,
  cex = 0.8,
  col = "darkgrey"
)

# === 染色体标签 ===
circos.trackPlotRegion(
  track.index = 1,
  bg.border    = NA,
  panel.fun    = function(x, y) {
    circos.text(
      CELL_META$xcenter,
      CELL_META$ylim[2] + mm_y(2),
      gsub("chr", "", CELL_META$sector.index),
      facing       = "clockwise",
      niceFacing   = TRUE,
      cex          = 0.8
    )
  }
)

# === 标题 & 图例 ===
title("Genomic Circos: CVD vs DR", col.main = "#444444", cex.main = 1.5)
legend(
  "bottom",
  legend = c("Exposure SNP", "Outcome SNP"),
  pch    = c(16, 17),
  col    = c(group_cols["exposure"], group_cols["outcome"]),
  pt.cex = 1,
  bty    = "n"
)

dev.off()