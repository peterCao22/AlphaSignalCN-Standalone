param(
  [string]$Dest = "D:\myCursor\AlphaSignalCN-Standalone",
  [switch]$IncludeLargeCsv
)

$ErrorActionPreference = "Stop"

# AlphaSignalCN-Standalone 导出脚本
# - 默认导出到：D:\myCursor\AlphaSignalCN-Standalone
# - 仅复制“核心代码+文档+模型+小体积DB”
# - 大体积 CSV 默认不拷贝（需要时可手动改开关）

function Copy-IfExists([string]$Src, [string]$Dst) {
  if (Test-Path $Src) {
    $item = Get-Item -LiteralPath $Src
    if ($item.PSIsContainer) {
      # 目录：复制“目录内容”到目标目录，避免出现 docs/docs、scripts/scripts 这类嵌套
      New-Item -ItemType Directory -Force -Path $Dst | Out-Null
      Copy-Item -Recurse -Force (Join-Path $Src "*") $Dst
    } else {
      # 文件：直接复制到目标文件路径
      New-Item -ItemType Directory -Force -Path (Split-Path $Dst) | Out-Null
      Copy-Item -Force $Src $Dst
    }
    Write-Host "[OK] Copied: $Src -> $Dst"
  } else {
    Write-Host "[WARN] Missing: $Src"
  }
}

function Remove-NestedDupIfExists([string]$Path) {
  if (Test-Path $Path) {
    Remove-Item -Recurse -Force $Path
    Write-Host "[OK] Removed nested duplicate: $Path"
  }
}

function Remove-PyCaches([string]$BaseDir) {
  if (-not (Test-Path $BaseDir)) { return }

  # 清理 Python 缓存：__pycache__ 目录 + *.pyc 文件
  Get-ChildItem -LiteralPath $BaseDir -Recurse -Force -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "__pycache__" } |
    ForEach-Object {
      Remove-Item -Recurse -Force $_.FullName
      Write-Host "[OK] Removed: $($_.FullName)"
    }

  Get-ChildItem -LiteralPath $BaseDir -Recurse -Force -File -ErrorAction SilentlyContinue -Include "*.pyc" |
    ForEach-Object {
      Remove-Item -Force $_.FullName
      Write-Host "[OK] Removed: $($_.FullName)"
    }
}

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $Here "..") | Select-Object -ExpandProperty Path

Write-Host "[INFO] Source root: $Root"
Write-Host "[INFO] Dest root  : $Dest"

# 0) 清理已存在的“重复嵌套目录”（通常由旧版 Copy-Item 目录复制方式导致）
Remove-NestedDupIfExists (Join-Path $Dest "docs\docs")
Remove-NestedDupIfExists (Join-Path $Dest "scripts\scripts")
Remove-NestedDupIfExists (Join-Path $Dest "models\models")
Remove-NestedDupIfExists (Join-Path $Dest "data\raw\raw")

# 1) 核心代码
Copy-IfExists (Join-Path $Root "predict_stock.py") (Join-Path $Dest "predict_stock.py")
Copy-IfExists (Join-Path $Root "scripts") (Join-Path $Dest "scripts")

# 1.1) 仅复制与 AlphaSignal-CN 预测链路相关的 stockainews 子集（不复制整个 stockainews）
# 说明：
# - stockainews 位于当前仓库根目录
# - 这里只复制 adapters/core/crawlers(部分)/services 等必要模块
$StockAiNewsSrc = Join-Path $Root "stockainews"
$StockAiNewsDst = Join-Path $Dest "stockainews"

Write-Host "[INFO] Repo root   : $Root"

$StockAiNewsWhitelist = @(
  "__init__.py",
  "default_config.py",
  "adapters",
  "core",
  (Join-Path "crawlers" "__init__.py"),
  (Join-Path "crawlers" "base_crawler.py"),
  (Join-Path "crawlers" "utils"),
  (Join-Path "crawlers" "legulegu"),
  (Join-Path "crawlers" "tonghuashun"),
  "services"
)

foreach ($rel in $StockAiNewsWhitelist) {
  Copy-IfExists (Join-Path $StockAiNewsSrc $rel) (Join-Path $StockAiNewsDst $rel)
}

# 2) 文档
Copy-IfExists (Join-Path $Root "docs") (Join-Path $Dest "docs")

# 3) 模型与特征名
Copy-IfExists (Join-Path $Root "models") (Join-Path $Dest "models")

# 4) DB（建议保留）
Copy-IfExists (Join-Path $Root "data\market_sentiment.db") (Join-Path $Dest "data\market_sentiment.db")
Copy-IfExists (Join-Path $Root "data\dragon_seats.db") (Join-Path $Dest "data\dragon_seats.db")
Copy-IfExists (Join-Path $Root "data\historical_patterns.db") (Join-Path $Dest "data\historical_patterns.db")

# 5) 小体积 raw（指数）
Copy-IfExists (Join-Path $Root "data\raw\index_bar1d.csv") (Join-Path $Dest "data\raw\index_bar1d.csv")

# 6) 可选：大体积 CSV
if ($IncludeLargeCsv) {
  Write-Host "[INFO] IncludeLargeCsv enabled, copying large CSV files..."
  Copy-IfExists (Join-Path $Root "data\raw") (Join-Path $Dest "data\raw")
}

# 7) 清理缓存文件（让 Standalone 更干净）
Remove-PyCaches $Dest

Write-Host "[OK] Export completed."

