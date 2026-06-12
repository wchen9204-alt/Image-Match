param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$ReportName = 'KEYPOINT_RIGID_TEST_REPORT_CN',
    [switch]$SkipRun,
    [switch]$SkipPdf
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$ReportRoot = Join-Path $ProjectRoot 'outputs\keypoint_rigid_report'
$ConfigRoot = Join-Path $ReportRoot '_configs'
$PipelineConfigRoot = Join-Path $ConfigRoot 'pipelines'
$ResultRoot = Join-Path $ReportRoot 'results'
$HtmlPath = Join-Path $ProjectRoot "$ReportName.html"
$PdfPath = Join-Path $ProjectRoot "$ReportName.pdf"
$ExePath = Join-Path $ProjectRoot 'build-mingw\bin\registration_app.exe'

$methods = @(
    [pscustomobject]@{ Label='akaze_rigid'; Display='AKAZE'; Keypoint='akaze.yaml' },
    [pscustomobject]@{ Label='brisk_rigid'; Display='BRISK'; Keypoint='brisk.yaml' },
    [pscustomobject]@{ Label='kaze_rigid'; Display='KAZE'; Keypoint='kaze.yaml' },
    [pscustomobject]@{ Label='orb_rigid'; Display='ORB'; Keypoint='orb.yaml' },
    [pscustomobject]@{ Label='sift_rigid'; Display='SIFT'; Keypoint='sift.yaml' },
    [pscustomobject]@{ Label='surf_rigid'; Display='SURF'; Keypoint='surf.yaml' }
)

function HtmlEscape {
    param([object]$Value)
    if ($null -eq $Value) { return '' }
    return [System.Net.WebUtility]::HtmlEncode([string]$Value)
}

function RelPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return '' }
    $full = [System.IO.Path]::GetFullPath($Path)
    $base = [System.IO.Path]::GetFullPath($ProjectRoot)
    if ($full.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)) {
        return (($full.Substring($base.Length).TrimStart('\', '/')) -replace '\\', '/')
    }
    return ($Path -replace '\\', '/')
}

function FileUri {
    param([string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path) -replace '\\', '/'
    return 'file:///' + ($full -replace ' ', '%20')
}

function FormatNumber {
    param([object]$Value, [int]$Digits = 3)
    if ($null -eq $Value) { return '-' }
    try {
        $number = [double]$Value
        if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { return '-' }
        if ($number -lt 0 -and $Digits -gt 0) { return '-' }
        return $number.ToString("F$Digits")
    } catch {
        return (HtmlEscape $Value)
    }
}

function PickProperty {
    param([object]$Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function MeanValue {
    param([object[]]$Items, [string]$PropertyName, [switch]$OnlyNonNegative)
    $values = @()
    foreach ($item in $Items) {
        $value = PickProperty $item $PropertyName
        if ($null -eq $value) { continue }
        try {
            $number = [double]$value
            if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { continue }
            if ($OnlyNonNegative -and $number -lt 0) { continue }
            $values += $number
        } catch {}
    }
    if ($values.Count -eq 0) { return $null }
    return ($values | Measure-Object -Average).Average
}

function New-Directory {
    param([string]$Path)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Create-RunConfigs {
    New-Directory $ConfigRoot
    New-Directory $PipelineConfigRoot

    foreach ($method in $methods) {
        $pipelineYaml = Join-Path $PipelineConfigRoot ($method.Label + '.yaml')
        $keypointPath = '../../../../configs/keypoint/' + $method.Keypoint
        $pipelineText = @"
name: $($method.Label)
method_family: keypoint
keypoint: $keypointPath
matcher: ../../../../configs/matcher/bf.yaml
filters:
  - ../../../../configs/filter/ratio_test.yaml
  - ../../../../configs/filter/cross_check.yaml
geometry: ../../../../configs/geometry/rigid.yaml
evaluator: ../../../../configs/evaluator/metrics.yaml

io:
  image1: ../../../../datasets/Test01/source.png
  image2: ../../../../datasets/Test01/target.png
  output_dir: ../results

visualization:
  draw_keypoints: true
  draw_matches: true
  draw_inliers_only: true
  max_matches_drawn: 120
  warp: true
  show_source_window: false
  show_target_window: false
  show_warped_window: false
  wait_key: 0

validation:
  warp_overlap:
    enabled: true
    min_iou: 0.10
    foreground_threshold: 10
  match_quality:
    enabled: true
    min_inliers: 4
    min_inlier_ratio: 0.10
    max_reproj_error: -1
  metric_quality:
    enabled: true
    min_ssim: 0.60
"@
        Write-Utf8NoBom $pipelineYaml $pipelineText

        $batchYaml = Join-Path $ConfigRoot ($method.Label + '_batch.yaml')
        $batchText = @"
name: $($method.Label)_batch
pipeline: pipelines/$($method.Label).yaml

dataset:
  root: ../../../datasets
  pattern_sources: [source, moving]
  pattern_targets: [target, reference]
  include: []

output:
  root: ../results
  save_visuals: true
  summary_csv: true
"@
        Write-Utf8NoBom $batchYaml $batchText
    }
}

function Invoke-KeypointRuns {
    if (-not (Test-Path -LiteralPath $ExePath)) {
        throw "registration_app not found: $ExePath"
    }

    foreach ($method in $methods) {
        $batchYaml = Join-Path $ConfigRoot ($method.Label + '_batch.yaml')
        Write-Host "Running $($method.Display): $batchYaml"
        & $ExePath $batchYaml
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "$($method.Display) finished with non-zero exit code: $LASTEXITCODE"
        }
    }
}

function Read-RunRecord {
    param([object]$Method, [string]$JsonPath)
    $json = Get-Content -LiteralPath $JsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $counts = PickProperty $json 'counts'
    $quality = PickProperty $json 'quality'
    $metrics = PickProperty $json 'metrics'
    $timings = PickProperty $json 'timings_ms'
    [pscustomobject]@{
        MethodLabel = $Method.Label
        MethodName = $Method.Display
        Sample = PickProperty $json 'sample_name'
        Status = PickProperty $json 'status'
        Success = ((PickProperty $json 'status') -eq 'OK')
        Message = PickProperty $json 'message'
        ResultDir = Split-Path $JsonPath -Parent
        SummaryPath = $JsonPath
        NumKeypoints1 = PickProperty $counts 'num_keypoints_first'
        NumKeypoints2 = PickProperty $counts 'num_keypoints_second'
        NumRawMatches = PickProperty $counts 'num_raw_matches'
        NumFilteredMatches = PickProperty $counts 'num_filtered_matches'
        NumInliers = PickProperty $counts 'num_inliers'
        InlierRatio = PickProperty $quality 'inlier_ratio'
        IoU = PickProperty $quality 'warp_overlap_iou'
        NMAD = PickProperty $quality 'warp_photometric_error'
        PSNR = PickProperty $metrics 'PSNR'
        SSIM = PickProperty $metrics 'SSIM'
        RMSE = PickProperty $metrics 'RMSE'
        LoadMs = PickProperty $timings 'load'
        ExtractMs = PickProperty $timings 'extract'
        MatchMs = PickProperty $timings 'match'
        FilterMs = PickProperty $timings 'filter'
        GeometryMs = PickProperty $timings 'geometry'
        WarpMs = PickProperty $timings 'warp'
        TotalMs = PickProperty $timings 'total'
    }
}

function Find-FirstImage {
    param([string]$Dir, [string[]]$Patterns)
    foreach ($pattern in $Patterns) {
        $file = Get-ChildItem -LiteralPath $Dir -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue |
            Sort-Object FullName |
            Select-Object -First 1
        if ($file) { return $file.FullName }
    }
    return ''
}

function ImageCellHtml {
    param([string]$Title, [string]$Path, [string]$FailureText)
    $titleHtml = HtmlEscape $Title
    if (-not [string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path)) {
        $src = HtmlEscape (FileUri $Path)
        return "<figure><figcaption>$titleHtml</figcaption><img loading=`"eager`" src=`"$src`" /></figure>"
    }
    return "<figure class=`"missing`"><figcaption>$titleHtml</figcaption><div>$([System.Net.WebUtility]::HtmlEncode($FailureText))</div></figure>"
}

function Build-Report {
    $records = @()
    foreach ($method in $methods) {
        $methodRoot = Join-Path (Join-Path (Join-Path $ResultRoot 'batch') 'keypoint') $method.Label
        $jsonFiles = @()
        if (Test-Path -LiteralPath $methodRoot) {
            $jsonFiles = @(Get-ChildItem -LiteralPath $methodRoot -Recurse -Filter run_summary.json | Sort-Object FullName)
        }
        foreach ($jsonFile in $jsonFiles) {
            $records += Read-RunRecord $method $jsonFile.FullName
        }
    }

    if ($records.Count -eq 0) {
        throw "No run_summary.json found under $ResultRoot"
    }

    $methodRows = foreach ($method in $methods) {
        $items = @($records | Where-Object { $_.MethodLabel -eq $method.Label })
        $ok = @($items | Where-Object { $_.Success }).Count
        $total = $items.Count
        $rate = if ($total -gt 0) { 100.0 * $ok / $total } else { 0.0 }
        '<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3} / {2}</td><td>{4}%</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td></tr>' -f `
            (HtmlEscape $method.Display),
            (HtmlEscape $method.Keypoint),
            $total,
            $ok,
            (FormatNumber $rate 1),
            (FormatNumber (MeanValue $items 'InlierRatio' -OnlyNonNegative) 3),
            (FormatNumber (MeanValue $items 'IoU' -OnlyNonNegative) 3),
            (FormatNumber (MeanValue $items 'SSIM' -OnlyNonNegative) 3),
            (FormatNumber (MeanValue $items 'TotalMs') 1)
    }

    $methodSections = foreach ($method in $methods) {
        $items = @($records | Where-Object { $_.MethodLabel -eq $method.Label } | Sort-Object Sample)
        $detailRows = foreach ($r in $items) {
            $statusClass = if ($r.Success) { 'ok' } else { 'fail' }
            $statusText = if ($r.Success) { '通过' } else { '未通过' }
            '<tr><td>{0}</td><td class="{1}">{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td><td>{9}</td><td>{10}</td><td>{11}</td><td>{12}</td><td>{13}</td><td>{14}</td><td>{15}</td></tr>' -f `
                (HtmlEscape $r.Sample),
                $statusClass,
                $statusText,
                (HtmlEscape $r.Message),
                (FormatNumber $r.NumKeypoints1 0),
                (FormatNumber $r.NumKeypoints2 0),
                (FormatNumber $r.NumRawMatches 0),
                (FormatNumber $r.NumFilteredMatches 0),
                (FormatNumber $r.NumInliers 0),
                (FormatNumber $r.InlierRatio 3),
                (FormatNumber $r.IoU 3),
                (FormatNumber $r.NMAD 4),
                (FormatNumber $r.SSIM 3),
                (FormatNumber $r.ExtractMs 1),
                (FormatNumber $r.MatchMs 1),
                (FormatNumber $r.TotalMs 1)
        }

        $imageBlocks = foreach ($r in $items) {
            $sourceKeypointsImage = Find-FirstImage $r.ResultDir @('*_source_keypoints.png')
            $targetKeypointsImage = Find-FirstImage $r.ResultDir @('*_target_keypoints.png')
            $inlierMatchImage = Find-FirstImage $r.ResultDir @('*_inlier_match.png', '*inlier_match*.png')
            $falseColorImage = Find-FirstImage $r.ResultDir @('*_false_color_overlay.png')
            $sampleTitle = HtmlEscape ($r.Sample + ' - ' + $method.Display)
            @"
<section class="sample-visual">
  <h4>$sampleTitle</h4>
  <div class="image-grid">
    $(ImageCellHtml '源图关键点检测图' $sourceKeypointsImage '生成失败：未找到源图关键点检测图')
    $(ImageCellHtml '目标图关键点检测图' $targetKeypointsImage '生成失败：未找到目标图关键点检测图')
    $(ImageCellHtml '内点匹配连线图' $inlierMatchImage '生成失败：未找到内点匹配连线图')
    $(ImageCellHtml '伪彩色配准误差叠加图' $falseColorImage '生成失败：未找到伪彩色配准误差叠加图')
  </div>
</section>
"@
        }

        @"
<section class="method-section page-break">
  <h2>$([System.Net.WebUtility]::HtmlEncode($method.Display))</h2>
  <h3>12 个测试用例过程信息统计表</h3>
  <table>
    <thead>
      <tr><th>样本</th><th>状态</th><th>说明</th><th>源关键点</th><th>目标关键点</th><th>原始匹配</th><th>过滤匹配</th><th>内点</th><th>内点率</th><th>IoU</th><th>NMAD</th><th>SSIM</th><th>提取 ms</th><th>匹配 ms</th><th>总耗时 ms</th></tr>
    </thead>
    <tbody>
      $($detailRows -join "`n")
    </tbody>
  </table>
  <h3>12 个测试用例可视化结果</h3>
  $($imageBlocks -join "`n")
</section>
"@
    }

    $css = @'
<style>
@page { size: A4 landscape; margin: 12mm; }
body { font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; color: #182033; line-height: 1.45; font-size: 12px; }
h1 { font-size: 24px; margin: 0 0 8px; }
h2 { font-size: 20px; margin: 24px 0 8px; border-bottom: 2px solid #1f4e79; padding-bottom: 4px; }
h3 { font-size: 15px; margin: 18px 0 8px; }
h4 { font-size: 13px; margin: 14px 0 6px; }
p { margin: 6px 0 10px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; table-layout: auto; }
th, td { border: 1px solid #cbd5e1; padding: 5px 6px; vertical-align: top; word-break: break-word; }
th { background: #eaf1f8; color: #102a43; font-weight: 700; }
tr:nth-child(even) td { background: #fafcff; }
.ok { color: #146c2e; font-weight: 700; }
.fail { color: #b42318; font-weight: 700; }
.page-break { break-before: page; }
.image-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; align-items: start; }
figure { margin: 0; border: 1px solid #d6dee8; padding: 6px; min-height: 118px; background: #fff; }
figcaption { font-weight: 700; margin-bottom: 5px; color: #334155; }
figure img { width: 100%; max-height: 190px; object-fit: contain; display: block; background: #f8fafc; }
figure.missing div { min-height: 84px; display: flex; align-items: center; justify-content: center; text-align: center; color: #9a3412; background: #fff7ed; border: 1px dashed #fdba74; padding: 8px; }
.note { color: #52606d; }
.sample-visual, figure { break-inside: avoid; page-break-inside: avoid; }
</style>
'@

    $html = @"
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>点特征法测试报告</title>
$css
</head>
<body>
<h1>点特征法测试报告</h1>
<h2>1. 六种方法 12 个用例测试结果与成功率</h2>
<table>
  <thead>
    <tr><th>方法</th><th>特征配置</th><th>用例数</th><th>成功数</th><th>成功率</th><th>平均内点率</th><th>平均 IoU</th><th>平均 SSIM</th><th>平均耗时 ms</th></tr>
  </thead>
  <tbody>
    $($methodRows -join "`n")
  </tbody>
</table>
$($methodSections -join "`n")
</body>
</html>
"@

    Write-Utf8NoBom $HtmlPath $html
    Write-Host "Generated HTML: $HtmlPath"
}

function Convert-HtmlToPdf {
    $edgeCandidates = @(
        'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        'C:\Program Files\Google\Chrome\Application\chrome.exe',
        'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
    )
    $browser = $edgeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $browser) {
        Write-Warning 'No Edge/Chrome executable found; skip PDF conversion.'
        return
    }

    $profile = Join-Path $ReportRoot '.edge-pdf-profile'
    if (Test-Path -LiteralPath $profile) {
        Remove-Item -LiteralPath $profile -Recurse -Force
    }
    $targetPdfPath = $PdfPath
    if (Test-Path -LiteralPath $targetPdfPath) {
        try {
            Remove-Item -LiteralPath $targetPdfPath -Force
        } catch {
            $targetPdfPath = Join-Path $ProjectRoot "$ReportName`_new.pdf"
            if (Test-Path -LiteralPath $targetPdfPath) {
                Remove-Item -LiteralPath $targetPdfPath -Force
            }
            Write-Warning "Original PDF is in use; writing to $targetPdfPath"
        }
    }
    $htmlUri = 'file:///' + (($HtmlPath -replace '\\', '/') -replace ' ', '%20')
    $args = @(
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        "--user-data-dir=$profile",
        '--no-first-run',
        '--disable-extensions',
        '--no-pdf-header-footer',
        '--print-to-pdf-no-header',
        "--print-to-pdf=$targetPdfPath",
        $htmlUri
    )
    $p = Start-Process -FilePath $browser -ArgumentList $args -Wait -PassThru -WindowStyle Hidden
    if ($p.ExitCode -ne 0) {
        Write-Warning "PDF conversion exited with code $($p.ExitCode)"
    }
    if (Test-Path -LiteralPath $profile) {
        Remove-Item -LiteralPath $profile -Recurse -Force
    }
    if (Test-Path -LiteralPath $targetPdfPath) {
        Write-Host "Generated PDF: $targetPdfPath"
    } else {
        Write-Warning "PDF was not generated: $targetPdfPath"
    }
}

Create-RunConfigs
if (-not $SkipRun) {
    Invoke-KeypointRuns
}
Build-Report
if (-not $SkipPdf) {
    Convert-HtmlToPdf
}
