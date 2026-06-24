param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$ReportName = 'DIRECT_FREQUENCY_INITIALIZER_TEST_REPORT_CN',
    [switch]$SkipPdf
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$CompareRoot = Join-Path $ProjectRoot 'outputs\compare\direct\frequency_initializers'
$HtmlPath = Join-Path $ProjectRoot "$ReportName.html"
$PdfPath = Join-Path $ProjectRoot "$ReportName.pdf"
$OutputReportDir = Join-Path $ProjectRoot 'reports\direct'

$CsvHeaders = @(
    'sample_name',
    'success',
    'message',
    'direct_confidence',
    'final_source',
    'initializer_inliers',
    'initializer_inlier_ratio',
    'initializer_coverage',
    'initializer_photometric_error',
    'overlap_containment',
    'source_coverage',
    'target_coverage',
    'bidirectional_coverage',
    'edge_iou',
    'photometric_error',
    'load_ms',
    'geometry_ms',
    'warp_ms',
    'total_ms',
    'metric_repeatability',
    'metric_inlier_ratio',
    'metric_reprojection_error'
)

$methods = @(
    [pscustomobject]@{ Label='frequency_akaze'; Display='AKAZE + Frequency' },
    [pscustomobject]@{ Label='frequency_brisk'; Display='BRISK + Frequency' },
    [pscustomobject]@{ Label='frequency_kaze'; Display='KAZE + Frequency' },
    [pscustomobject]@{ Label='frequency_orb'; Display='ORB + Frequency' },
    [pscustomobject]@{ Label='frequency_sift'; Display='SIFT + Frequency' },
    [pscustomobject]@{ Label='frequency_surf'; Display='SURF + Frequency' }
)

$Text = @{
    ReportTitle = '&#30452;&#25509;&#27861;&#39057;&#22495;&#21021;&#22987;&#21270;&#23545;&#27604;&#27979;&#35797;&#25253;&#21578;'
    Section1 = '1. 6 &#32452;&#21021;&#22987;&#21270;&#22120;&#30340;&#24635;&#27979;&#35797;&#32467;&#26524;&#21644;&#25104;&#21151;&#29575;'
    Section2 = '2. 12 &#20010;&#29992;&#20363;&#35814;&#32454;&#27979;&#35797;&#32467;&#26524;&#19982;&#27719;&#24635;&#20449;&#24687;'
    Section3 = '3. &#21487;&#35270;&#21270;&#32467;&#26524;'
    Sample = '&#29992;&#20363;'
    Status = '&#29366;&#24577;'
    Passed = '&#36890;&#36807;'
    Failed = '&#26410;&#36890;&#36807;'
    Message = '&#35828;&#26126;'
    DirectConfidence = '&#30452;&#25509;&#27861;&#32622;&#20449;&#24230;'
    FinalSource = '&#26368;&#32456;&#37319;&#29992;&#26469;&#28304;'
    InitInliers = '&#21021;&#22987;&#20540;&#20869;&#28857;&#25968;'
    InitInlierRatio = '&#21021;&#22987;&#20540;&#20869;&#28857;&#29575;'
    InitCoverage = '&#21021;&#22987;&#20540;&#31354;&#38388;&#35206;&#30422;&#29575;'
    InitPhotometric = '&#21021;&#22987;&#20540;&#20809;&#24230;&#35823;&#24046;'
    Containment = '&#37325;&#21472;&#21253;&#21547;&#29575;'
    BiCoverage = '&#21452;&#21521;&#35206;&#30422;&#29575;'
    EdgeIoU = '&#36793;&#32536;&#23545;&#40784; IoU'
    Nmad = '&#20809;&#24230;&#35823;&#24046;'
    GeometryMs = '&#20960;&#20309;&#38454;&#27573;&#32791;&#26102; ms'
    TotalMs = '&#24635;&#32791;&#26102; ms'
    CaseCount = '&#29992;&#20363;&#25968;'
    SuccessCount = '&#25104;&#21151;&#25968;'
    SuccessRate = '&#25104;&#21151;&#29575;'
    AvgConfidence = '&#24179;&#22343;&#30452;&#25509;&#27861;&#32622;&#20449;&#24230;'
    AvgEdgeIoU = '&#24179;&#22343;&#36793;&#32536; IoU'
    AvgTotalMs = '&#24179;&#22343;&#24635;&#32791;&#26102; ms'
    Method = '&#26041;&#27861;'
    FinalOverlayFig = '&#26368;&#32456;&#20266;&#33394;&#24425;&#37197;&#20934;&#35823;&#24046;&#21472;&#21152;&#22270;'
    DirectOverlayFig = '&#30452;&#25509;&#27861;&#20266;&#33394;&#24425;&#21472;&#21152;&#22270;'
    InitOverlayFig = '&#21021;&#22987;&#20540;&#20266;&#33394;&#24425;&#21472;&#21152;&#22270;'
    BlendFig = 'Blend'
    MissingFinalOverlay = '&#26410;&#25214;&#21040;&#26368;&#32456;&#20266;&#33394;&#24425;&#37197;&#20934;&#35823;&#24046;&#21472;&#21152;&#22270;'
    MissingDirectOverlay = '&#26410;&#25214;&#21040;&#30452;&#25509;&#27861;&#20266;&#33394;&#24425;&#21472;&#21152;&#22270;'
    MissingInitOverlay = '&#26410;&#25214;&#21040;&#21021;&#22987;&#20540;&#20266;&#33394;&#24425;&#21472;&#21152;&#22270;'
    MissingBlend = '&#26410;&#25214;&#21040; blend &#21472;&#21152;&#22270;'
}

function HtmlEscape {
    param([object]$Value)
    if ($null -eq $Value) { return '' }
    return [System.Net.WebUtility]::HtmlEncode([string]$Value)
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
        $text = [string]$Value
        if ([string]::IsNullOrWhiteSpace($text)) { return '-' }
        $number = [double]$text
        if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { return '-' }
        if ($number -lt 0 -and $Digits -gt 0) { return '-' }
        return $number.ToString("F$Digits")
    } catch {
        return (HtmlEscape $Value)
    }
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$TextValue)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $TextValue, $encoding)
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

function Find-DirectOverlayImage {
    param([string]$Dir)
    $file = Get-ChildItem -LiteralPath $Dir -Recurse -File -Filter '*_false_color_overlay.png' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike '*_initializer_false_color_overlay.png' } |
        Sort-Object FullName |
        Select-Object -First 1
    if ($file) { return $file.FullName }
    return ''
}

function ImageCellHtml {
    param([string]$TitleHtml, [string]$Path, [string]$FailureHtml)
    if (-not [string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path)) {
        $src = HtmlEscape (FileUri $Path)
        return "<figure><figcaption>$TitleHtml</figcaption><img loading=`"eager`" src=`"$src`" /></figure>"
    }
    return "<figure class=`"missing`"><figcaption>$TitleHtml</figcaption><div>$FailureHtml</div></figure>"
}

function Import-CompareCsv {
    param([string]$CsvPath)
    $lines = Get-Content -LiteralPath $CsvPath -Encoding UTF8
    if ($lines.Count -le 1) { return @() }
    $dataLines = $lines | Select-Object -Skip 1
    return @($dataLines | ConvertFrom-Csv -Header $CsvHeaders)
}

function Read-MethodRows {
    param([pscustomobject]$Method)
    $summaryCsv = Join-Path (Join-Path $CompareRoot $Method.Label) 'summary.csv'
    if (-not (Test-Path -LiteralPath $summaryCsv)) {
        throw "Missing summary.csv: $summaryCsv"
    }

    $rows = Import-CompareCsv $summaryCsv
    foreach ($row in $rows) {
        $resultDir = Join-Path (Join-Path $CompareRoot $Method.Label) $row.sample_name
        [pscustomobject]@{
            MethodLabel = $Method.Label
            MethodName = $Method.Display
            Sample = $row.sample_name
            Success = ($row.success -eq '1')
            StatusHtml = if ($row.success -eq '1') { $Text.Passed } else { $Text.Failed }
            Message = $row.message
            ResultDir = $resultDir
            SummaryCsv = $summaryCsv
            DirectConfidence = $row.direct_confidence
            FinalSource = $row.final_source
            InitInliers = $row.initializer_inliers
            InitInlierRatio = $row.initializer_inlier_ratio
            InitCoverage = $row.initializer_coverage
            InitPhotometric = $row.initializer_photometric_error
            Containment = $row.overlap_containment
            BidirectionalCoverage = $row.bidirectional_coverage
            EdgeIoU = $row.edge_iou
            NMAD = $row.photometric_error
            GeometryMs = $row.geometry_ms
            TotalMs = $row.total_ms
        }
    }
}

function MeanValue {
    param([object[]]$Items, [string]$PropertyName)
    $values = @()
    foreach ($item in $Items) {
        $value = $item.$PropertyName
        if ($null -eq $value -or [string]::IsNullOrWhiteSpace([string]$value)) { continue }
        try {
            $number = [double]$value
            if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { continue }
            if ($number -lt 0) { continue }
            $values += $number
        } catch {}
    }
    if ($values.Count -eq 0) { return $null }
    return ($values | Measure-Object -Average).Average
}

function Build-Report {
    if (-not (Test-Path -LiteralPath $CompareRoot)) {
        throw "Compare output directory not found: $CompareRoot"
    }

    $records = @()
    foreach ($method in $methods) {
        $records += @(Read-MethodRows $method)
    }

    $overallRows = foreach ($method in $methods) {
        $items = @($records | Where-Object { $_.MethodLabel -eq $method.Label })
        $ok = @($items | Where-Object { $_.Success }).Count
        $total = $items.Count
        $rate = if ($total -gt 0) { 100.0 * $ok / $total } else { 0.0 }
        '<tr><td>{0}</td><td>{1}</td><td>{2} / {1}</td><td>{3}%</td><td>{4}</td><td>{5}</td><td>{6}</td></tr>' -f `
            (HtmlEscape $method.Display),
            $total,
            $ok,
            (FormatNumber $rate 1),
            (FormatNumber (MeanValue $items 'DirectConfidence') 3),
            (FormatNumber (MeanValue $items 'EdgeIoU') 3),
            (FormatNumber (MeanValue $items 'TotalMs') 1)
    }

    $methodSections = foreach ($method in $methods) {
        $items = @($records | Where-Object { $_.MethodLabel -eq $method.Label } | Sort-Object Sample)
        $detailRows = foreach ($r in $items) {
            $statusClass = if ($r.Success) { 'ok' } else { 'fail' }
            '<tr><td>{0}</td><td class="{1}">{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td><td>{9}</td><td>{10}</td><td>{11}</td><td>{12}</td><td>{13}</td><td>{14}</td><td>{15}</td></tr>' -f `
                (HtmlEscape $r.Sample),
                $statusClass,
                $r.StatusHtml,
                (HtmlEscape $r.Message),
                (FormatNumber $r.DirectConfidence 3),
                (HtmlEscape $r.FinalSource),
                (FormatNumber $r.InitInliers 0),
                (FormatNumber $r.InitInlierRatio 3),
                (FormatNumber $r.InitCoverage 3),
                (FormatNumber $r.InitPhotometric 4),
                (FormatNumber $r.Containment 3),
                (FormatNumber $r.BidirectionalCoverage 3),
                (FormatNumber $r.EdgeIoU 3),
                (FormatNumber $r.NMAD 4),
                (FormatNumber $r.GeometryMs 1),
                (FormatNumber $r.TotalMs 1)
        }

        $imageBlocks = foreach ($r in $items) {
            $finalOverlayImage = Find-FirstImage $r.ResultDir @('*_final_false_color_overlay.png')
            $directOverlayImage = Find-DirectOverlayImage $r.ResultDir
            $initializerOverlayImage = Find-FirstImage $r.ResultDir @('*_initializer_false_color_overlay.png')
            $blendImage = Find-FirstImage $r.ResultDir @('*_blend.png')
            $sampleTitle = HtmlEscape ($r.Sample + ' - ' + $method.Display)
@"
<section class="sample-visual">
  <h4>$sampleTitle</h4>
  <div class="image-grid">
    $(ImageCellHtml $Text.FinalOverlayFig $finalOverlayImage $Text.MissingFinalOverlay)
    $(ImageCellHtml $Text.DirectOverlayFig $directOverlayImage $Text.MissingDirectOverlay)
    $(ImageCellHtml $Text.InitOverlayFig $initializerOverlayImage $Text.MissingInitOverlay)
    $(ImageCellHtml $Text.BlendFig $blendImage $Text.MissingBlend)
  </div>
</section>
"@
        }

@"
<section class="method-section page-break">
  <h2>$([System.Net.WebUtility]::HtmlEncode($method.Display))</h2>
  <h3>$($Text.Section2)</h3>
  <table>
    <thead>
      <tr><th>$($Text.Sample)</th><th>$($Text.Status)</th><th>$($Text.Message)</th><th>$($Text.DirectConfidence)</th><th>$($Text.FinalSource)</th><th>$($Text.InitInliers)</th><th>$($Text.InitInlierRatio)</th><th>$($Text.InitCoverage)</th><th>$($Text.InitPhotometric)</th><th>$($Text.Containment)</th><th>$($Text.BiCoverage)</th><th>$($Text.EdgeIoU)</th><th>$($Text.Nmad)</th><th>$($Text.GeometryMs)</th><th>$($Text.TotalMs)</th></tr>
    </thead>
    <tbody>
      $($detailRows -join "`n")
    </tbody>
  </table>
  <h3>$($Text.Section3)</h3>
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
<title>$($Text.ReportTitle)</title>
$css
</head>
<body>
<h1>$($Text.ReportTitle)</h1>
<h2>$($Text.Section1)</h2>
<table>
  <thead>
    <tr><th>$($Text.Method)</th><th>$($Text.CaseCount)</th><th>$($Text.SuccessCount)</th><th>$($Text.SuccessRate)</th><th>$($Text.AvgConfidence)</th><th>$($Text.AvgEdgeIoU)</th><th>$($Text.AvgTotalMs)</th></tr>
  </thead>
  <tbody>
    $($overallRows -join "`n")
  </tbody>
</table>
$($methodSections -join "`n")
</body>
</html>
"@

    New-Item -ItemType Directory -Path $OutputReportDir -Force | Out-Null
    Write-Utf8NoBom $HtmlPath $html
    Write-Utf8NoBom (Join-Path $OutputReportDir "$ReportName.html") $html
    Write-Host "Generated HTML: $HtmlPath"
}

function Convert-HtmlToPdf {
    $browserCandidates = @(
        'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        'C:\Program Files\Google\Chrome\Application\chrome.exe',
        'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
    )
    $browser = $browserCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $browser) {
        Write-Warning 'No Edge/Chrome executable found; skip PDF conversion.'
        return
    }

    $profile = Join-Path $ProjectRoot 'outputs\compare_direct_frequency_report\.edge-pdf-profile'
    if (Test-Path -LiteralPath $profile) {
        Remove-Item -LiteralPath $profile -Recurse -Force
    }
    $targetPdfPath = $PdfPath
    if (Test-Path -LiteralPath $targetPdfPath) {
        Remove-Item -LiteralPath $targetPdfPath -Force
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
        Copy-Item -LiteralPath $targetPdfPath -Destination (Join-Path $OutputReportDir "$ReportName.pdf") -Force
        Write-Host "Generated PDF: $targetPdfPath"
    }
}

Build-Report
if (-not $SkipPdf) {
    Convert-HtmlToPdf
}
