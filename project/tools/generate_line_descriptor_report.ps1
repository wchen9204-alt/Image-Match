param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$ReportName = 'LINE_DESCRIPTOR_TEST_REPORT_CN',
    [switch]$SkipRun,
    [switch]$SkipPdf
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$ReportRoot = Join-Path $ProjectRoot 'outputs\line_descriptor_report'
$ConfigRoot = Join-Path $ReportRoot '_configs'
$StructureConfigRoot = Join-Path $ConfigRoot 'structures'
$PipelineConfigRoot = Join-Path $ConfigRoot 'pipelines'
$ResultRoot = Join-Path $ReportRoot 'results'
$ReportOutputRoot = Join-Path $ProjectRoot 'reports\line_descriptor'
$HtmlPath = Join-Path $ReportOutputRoot "$ReportName.html"
$PdfPath = Join-Path $ReportOutputRoot "$ReportName.pdf"
$ExePath = Join-Path $ProjectRoot 'build-mingw\bin\registration_app.exe'

$methods = @()
foreach ($extractor in @('HOUGH_LINES_P', 'HOUGH_LINES', 'LSD', 'FLD')) {
    $extractorLabel = $extractor.ToLower()
    $methods += [pscustomobject]@{
        Label = $extractorLabel + '_line_segment'
        Display = $extractor + ' + Line Segment'
        Extractor = $extractor
        Association = 'LINE_SEGMENT'
        Descriptor = 'LINE_SEGMENT'
    }
    foreach ($descriptor in @('LBD', 'MSLD', 'LINE_SIFT')) {
        $methods += [pscustomobject]@{
            Label = $extractorLabel + '_' + $descriptor.ToLower()
            Display = $extractor + ' + ' + $descriptor
            Extractor = $extractor
            Association = 'LINE_DESCRIPTOR'
            Descriptor = $descriptor
        }
    }
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
    New-Directory $StructureConfigRoot
    New-Directory $PipelineConfigRoot

    foreach ($method in $methods) {
        $structureYaml = Join-Path $StructureConfigRoot ($method.Label + '.yaml')
        $structureText = @"
type: LINE

extractor:
  method: $($method.Extractor)
  params:
    cannyThreshold1: 50.0
    cannyThreshold2: 150.0
    apertureSize: 3
    rho: 1.0
    thetaDegrees: 1.0
    threshold: 50
    maxLines: 300
    minLineLength: 30.0
    maxLineGap: 10.0
    lineThickness: 1
    deduplicateLines: false
    duplicateAngleDeg: 3.0
    duplicateDistance: 8.0
    lsd:
      refine: 1
      scale: 0.8
      sigmaScale: 0.6
      quant: 2.0
      angTh: 22.5
      logEps: 0.0
      densityTh: 0.7
      nBins: 1024
      detectorScale: 2
      detectorNumOctaves: 2
    fld:
      lengthThreshold: 10
      distanceThreshold: 1.414213562
      cannyThreshold1: 50.0
      cannyThreshold2: 50.0
      cannyApertureSize: 3
      doMerge: false

association:
  method: $($method.Association)
  params:
    line_descriptor:
      descriptor: $($method.Descriptor)
      matcher: BF
      match_mode: MATCH
      knn_k: 2
      match_radius: 50.0
      min_matches: 2
      geometric_filter: true
      angle_threshold_deg: 30.0
      min_length_ratio: 0.30
      shift_consistency_threshold: 30.0
      msld_band_width: 12
      msld_strips: 9
      linesift_band_width: 20
      linesift_strips: 4
      linesift_bands: 4
      linesift_bins: 8
    line_segment:
      angleThresholdDeg: 10.0
      minLengthRatio: 0.60
      maxShiftDistance: 100000.0
      shiftConsistencyThreshold: 20.0
      minMatches: 2
      maxCandidatesPerLine: 5

estimation:
  responseThreshold: 0.01
"@
        Write-Utf8NoBom $structureYaml $structureText

        $pipelineYaml = Join-Path $PipelineConfigRoot ($method.Label + '.yaml')
        $pipelineText = @"
name: $($method.Label)
method_family: structure
structure: ../structures/$($method.Label).yaml
geometry: ../../../../configs/geometry/rigid.yaml
evaluator: ../../../../configs/evaluator/metrics.yaml
filters: []

io:
  image1: ../../../../datasets/Test01/source.png
  image2: ../../../../datasets/Test01/target.png
  output_dir: ../results

visualization:
  draw_matches: true
  draw_inliers_only: true
  max_matches_drawn: 100
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
  structure_overlap:
    enabled: true
    min_iou: 0.12
    foreground_threshold: 0
    dilate_size: 7
  metric_quality:
    enabled: true
    min_ssim: 0.66
  photometric:
    enabled: false
    max_nmad: 0.15
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

function Invoke-LineRuns {
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
        Extractor = $Method.Extractor
        Association = $Method.Association
        Descriptor = $Method.Descriptor
        Sample = PickProperty $json 'sample_name'
        Status = PickProperty $json 'status'
        Success = ((PickProperty $json 'status') -eq 'OK')
        Message = PickProperty $json 'message'
        ResultDir = Split-Path $JsonPath -Parent
        NumStructures1 = PickProperty $counts 'num_structures_first'
        NumStructures2 = PickProperty $counts 'num_structures_second'
        NumRawMatches = PickProperty $counts 'num_raw_matches'
        NumFilteredMatches = PickProperty $counts 'num_filtered_matches'
        NumInliers = PickProperty $counts 'num_inliers'
        InlierRatio = PickProperty $quality 'inlier_ratio'
        IoU = PickProperty $quality 'warp_overlap_iou'
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
    $titleHtml = [string]$Title
    $failureHtml = [string]$FailureText
    if (-not [string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path)) {
        $src = HtmlEscape (FileUri $Path)
        return "<figure><figcaption>$titleHtml</figcaption><img loading=`"eager`" src=`"$src`" /></figure>"
    }
    return "<figure class=`"missing`"><figcaption>$titleHtml</figcaption><div>$failureHtml</div></figure>"
}

function Build-Report {
    New-Directory $ReportOutputRoot
    $records = @()
    foreach ($method in $methods) {
        $methodRoot = Join-Path (Join-Path (Join-Path $ResultRoot 'batch') 'structure') $method.Label
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
        '<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5} / {4}</td><td>{6}%</td><td>{7}</td><td>{8}</td><td>{9}</td><td>{10}</td></tr>' -f `
            (HtmlEscape $method.Display),
            (HtmlEscape $method.Extractor),
            (HtmlEscape $method.Association),
            (HtmlEscape $method.Descriptor),
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
            $statusText = if ($r.Success) { '&#36890;&#36807;' } else { '&#26410;&#36890;&#36807;' }
            '<tr><td>{0}</td><td class="{1}">{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td><td>{9}</td><td>{10}</td><td>{11}</td><td>{12}</td><td>{13}</td><td>{14}</td></tr>' -f `
                (HtmlEscape $r.Sample),
                $statusClass,
                $statusText,
                (HtmlEscape $r.Message),
                (FormatNumber $r.NumStructures1 0),
                (FormatNumber $r.NumStructures2 0),
                (FormatNumber $r.NumRawMatches 0),
                (FormatNumber $r.NumFilteredMatches 0),
                (FormatNumber $r.NumInliers 0),
                (FormatNumber $r.InlierRatio 3),
                (FormatNumber $r.IoU 3),
                (FormatNumber $r.SSIM 3),
                (FormatNumber $r.ExtractMs 1),
                (FormatNumber $r.MatchMs 1),
                (FormatNumber $r.TotalMs 1)
        }

        $imageBlocks = foreach ($r in $items) {
            $sourceStructureImage = Find-FirstImage $r.ResultDir @('*_source_structure.png')
            $targetStructureImage = Find-FirstImage $r.ResultDir @('*_target_structure.png')
            $matchImage = Find-FirstImage $r.ResultDir @('*_structure_matches.png')
            $falseColorImage = Find-FirstImage $r.ResultDir @('*_false_color_overlay.png')
            $sampleTitle = HtmlEscape ($r.Sample + ' - ' + $method.Display)
            @"
<section class="sample-visual">
  <h4>$sampleTitle</h4>
  <div class="image-grid">
    $(ImageCellHtml '&#28304;&#22270;&#30452;&#32447;&#32467;&#26500;&#22270;' $sourceStructureImage '&#29983;&#25104;&#22833;&#36133;&#65306;&#26410;&#25214;&#21040;&#28304;&#22270; structures &#22270;')
    $(ImageCellHtml '&#30446;&#26631;&#22270;&#30452;&#32447;&#32467;&#26500;&#22270;' $targetStructureImage '&#29983;&#25104;&#22833;&#36133;&#65306;&#26410;&#25214;&#21040;&#30446;&#26631;&#22270; structures &#22270;')
    $(ImageCellHtml '&#30452;&#32447;&#21305;&#37197;&#36830;&#32447;&#22270;' $matchImage '&#29983;&#25104;&#22833;&#36133;&#65306;&#26410;&#25214;&#21040; matches &#22270;')
    $(ImageCellHtml '&#20266;&#24425;&#33394;&#37197;&#20934;&#35823;&#24046;&#21472;&#21152;&#22270;' $falseColorImage '&#29983;&#25104;&#22833;&#36133;&#65306;&#26410;&#25214;&#21040; false_color_overlay &#22270;')
  </div>
</section>
"@
        }

        @"
<section class="method-section page-break">
  <h2>$([System.Net.WebUtility]::HtmlEncode($method.Display))</h2>
  <h3>12 &#20010;&#27979;&#35797;&#29992;&#20363;&#36807;&#31243;&#20449;&#24687;&#32479;&#35745;&#34920;</h3>
  <table>
    <thead>
      <tr><th>&#26679;&#26412;</th><th>&#29366;&#24577;</th><th>&#35828;&#26126;</th><th>&#28304;&#30452;&#32447;&#25968;</th><th>&#30446;&#26631;&#30452;&#32447;&#25968;</th><th>&#20505;&#36873;&#21305;&#37197;</th><th>&#36807;&#28388;&#21305;&#37197;</th><th>&#20869;&#28857;</th><th>&#20869;&#28857;&#29575;</th><th>IoU</th><th>SSIM</th><th>&#25552;&#21462; ms</th><th>&#21305;&#37197; ms</th><th>&#24635;&#32791;&#26102; ms</th></tr>
    </thead>
    <tbody>
      $($detailRows -join "`n")
    </tbody>
  </table>
  <h3>12 &#20010;&#27979;&#35797;&#29992;&#20363;&#21487;&#35270;&#21270;&#32467;&#26524;</h3>
  $($imageBlocks -join "`n")
</section>
"@
    }

    $css = @'
<style>
@page { size: A4 landscape; margin: 12mm; }
body { font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; color: #182033; line-height: 1.45; font-size: 12px; }
h1 { font-size: 24px; margin: 0 0 8px; }
h2 { font-size: 20px; margin: 24px 0 8px; border-bottom: 2px solid #375f2d; padding-bottom: 4px; }
h3 { font-size: 15px; margin: 18px 0 8px; }
h4 { font-size: 13px; margin: 14px 0 6px; }
p { margin: 6px 0 10px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; table-layout: auto; }
th, td { border: 1px solid #cbd5e1; padding: 5px 6px; vertical-align: top; word-break: break-word; }
th { background: #edf7ea; color: #17351b; font-weight: 700; }
tr:nth-child(even) td { background: #fbfdf9; }
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
<title>&#30452;&#32447;&#26816;&#27979;&#19982;&#25551;&#36848;&#23376;&#27861;&#27979;&#35797;&#25253;&#21578;</title>
$css
</head>
<body>
<h1>&#30452;&#32447;&#26816;&#27979;&#19982;&#25551;&#36848;&#23376;&#27861;&#27979;&#35797;&#25253;&#21578;</h1>
<p class="note">&#26412;&#25253;&#21578;&#27979;&#35797; 4 &#31181;&#30452;&#32447;&#26816;&#27979;&#26041;&#27861;&#65288;HOUGH_LINES_P&#12289;HOUGH_LINES&#12289;LSD&#12289;FLD&#65289;&#19982; LINE_SEGMENT / LBD / MSLD / LINE_SIFT &#32452;&#21512;&#65292;&#20960;&#20309;&#20272;&#35745;&#32479;&#19968;&#20351;&#29992; RIGID&#65307;&#25253;&#21578;&#22270;&#29255;&#23637;&#31034; structures&#12289;matches&#12289;false_color_overlay &#19977;&#31867;&#36755;&#20986;&#12290;</p>
<h2>1. &#30452;&#32447;&#26816;&#27979;&#19982;&#25551;&#36848;&#23376;&#32452;&#21512; 12 &#20010;&#29992;&#20363;&#27979;&#35797;&#32467;&#26524;&#19982;&#25104;&#21151;&#29575;</h2>
<table>
  <thead>
    <tr><th>&#26041;&#27861;</th><th>&#26816;&#27979;&#22120;</th><th>&#20851;&#32852;&#26041;&#27861;</th><th>&#25551;&#36848;&#23376;</th><th>&#29992;&#20363;&#25968;</th><th>&#25104;&#21151;&#25968;</th><th>&#25104;&#21151;&#29575;</th><th>&#24179;&#22343;&#20869;&#28857;&#29575;</th><th>&#24179;&#22343; IoU</th><th>&#24179;&#22343; SSIM</th><th>&#24179;&#22343;&#32791;&#26102; ms</th></tr>
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
    New-Directory $ReportOutputRoot
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
            $targetPdfPath = Join-Path $ReportOutputRoot "$ReportName`_new.pdf"
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
    Invoke-LineRuns
}
Build-Report
if (-not $SkipPdf) {
    Convert-HtmlToPdf
}
