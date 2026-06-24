param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$OutputFile = ''
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($OutputFile)) {
    $OutputFile = Join-Path $ProjectRoot 'reports\direct\TEST_CASES_REPORT_CN.html'
}

$BatchRoot = Join-Path $ProjectRoot 'outputs\batch'
$DatasetRoot = Join-Path $ProjectRoot 'datasets'

$SampleNameColumn = [string]([char]0x6837)+[char]0x672C+[char]0x540D
$SuccessColumn = [string]([char]0x662F)+[char]0x5426+[char]0x6210+[char]0x529F
$MessageColumn = [string]([char]0x7ED3)+[char]0x679C+[char]0x8BF4+[char]0x660E
$KeypointFirstColumn = [string]([char]0x5173)+[char]0x952E+[char]0x70B9+[char]0x6570+[char]0x005F+[char]0x7B2C+[char]0x4E00+[char]0x5F20
$KeypointSecondColumn = [string]([char]0x5173)+[char]0x952E+[char]0x70B9+[char]0x6570+[char]0x005F+[char]0x7B2C+[char]0x4E8C+[char]0x5F20
$StructureFirstColumn = [string]([char]0x7ED3)+[char]0x6784+[char]0x6570+[char]0x005F+[char]0x7B2C+[char]0x4E00+[char]0x5F20
$StructureSecondColumn = [string]([char]0x7ED3)+[char]0x6784+[char]0x6570+[char]0x005F+[char]0x7B2C+[char]0x4E8C+[char]0x5F20
$LearningFirstColumn = [string]([char]0x5B66)+[char]0x4E60+[char]0x70B9+[char]0x6570+[char]0x005F+[char]0x7B2C+[char]0x4E00+[char]0x5F20
$LearningSecondColumn = [string]([char]0x5B66)+[char]0x4E60+[char]0x70B9+[char]0x6570+[char]0x005F+[char]0x7B2C+[char]0x4E8C+[char]0x5F20
$DirectConfidenceColumn = [string]([char]0x76F4)+[char]0x63A5+[char]0x6CD5+[char]0x7F6E+[char]0x4FE1+[char]0x5EA6
$FinalSourceColumn = [string]([char]0x6700)+[char]0x7EC8+[char]0x91C7+[char]0x7528+[char]0x6765+[char]0x6E90
$InitializerInliersColumn = [string]([char]0x521D)+[char]0x59CB+[char]0x503C+[char]0x5185+[char]0x70B9+[char]0x6570
$InitializerInlierRatioColumn = [string]([char]0x521D)+[char]0x59CB+[char]0x503C+[char]0x5185+[char]0x70B9+[char]0x7387
$InitializerCoverageColumn = [string]([char]0x521D)+[char]0x59CB+[char]0x503C+[char]0x7A7A+[char]0x95F4+[char]0x8986+[char]0x76D6+[char]0x7387
$InitializerPhotometricColumn = [string]([char]0x521D)+[char]0x59CB+[char]0x503C+[char]0x5149+[char]0x5EA6+[char]0x8BEF+[char]0x5DEE
$ContainmentColumn = [string]([char]0x91CD)+[char]0x53E0+[char]0x5305+[char]0x542B+[char]0x7387
$SourceCoverageColumn = [string]([char]0x6E90)+[char]0x56FE+[char]0x8986+[char]0x76D6+[char]0x7387
$TargetCoverageColumn = [string]([char]0x76EE)+[char]0x6807+[char]0x56FE+[char]0x8986+[char]0x76D6+[char]0x7387
$BidirectionalCoverageColumn = [string]([char]0x53CC)+[char]0x5411+[char]0x8986+[char]0x76D6+[char]0x7387
$EdgeAlignmentColumn = [string]([char]0x8FB9)+[char]0x7F18+[char]0x5BF9+[char]0x9F50+[char]0x0049+[char]0x006F+[char]0x0055
$PhotometricColumn = [string]([char]0x5149)+[char]0x5EA6+[char]0x8BEF+[char]0x5DEE
$LoadMsColumn = [string]([char]0x52A0)+[char]0x8F7D+[char]0x8017+[char]0x65F6+[char]0x005F+[char]0x006D+[char]0x0073
$GeometryMsColumn = [string]([char]0x51E0)+[char]0x4F55+[char]0x9636+[char]0x6BB5+[char]0x8017+[char]0x65F6+[char]0x005F+[char]0x006D+[char]0x0073
$WarpMsColumn = [string]([char]0x53D8)+[char]0x6362+[char]0x8017+[char]0x65F6+[char]0x005F+[char]0x006D+[char]0x0073
$TotalMsColumn = [string]([char]0x603B)+[char]0x8017+[char]0x65F6+[char]0x005F+[char]0x006D+[char]0x0073

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

function PickProperty {
    param([object]$Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function PickFirst {
    param([object]$Object, [string[]]$Names)
    foreach ($name in $Names) {
        $value = PickProperty $Object $name
        if ($null -ne $value -and -not [string]::IsNullOrWhiteSpace([string]$value)) {
            return $value
        }
    }
    return $null
}

function HasColumn {
    param([object]$Object, [string]$Name)
    return $null -ne $Object -and $null -ne $Object.PSObject.Properties[$Name]
}

function New-UniqueCsvHeader {
    param([string[]]$Headers)
    $seen = @{}
    $unique = @()
    foreach ($header in $Headers) {
        $name = if ([string]::IsNullOrWhiteSpace($header)) { 'column' } else { $header }
        $key = $name.ToLowerInvariant()
        if ($seen.ContainsKey($key)) {
            $candidate = if ($name -notlike 'metric_*' -and $name -notlike '指标_*') {
                'metric_' + $name
            } else {
                $name + '_2'
            }
            $candidateKey = $candidate.ToLowerInvariant()
            $suffix = 2
            while ($seen.ContainsKey($candidateKey)) {
                $candidate = $name + '_' + $suffix
                $candidateKey = $candidate.ToLowerInvariant()
                $suffix += 1
            }
            $name = $candidate
            $key = $candidateKey
        }
        $seen[$key] = $true
        $unique += $name
    }
    return $unique
}

function Read-CsvRowsUnique {
    param([string]$Path)
    Add-Type -AssemblyName Microsoft.VisualBasic
    $parser = [Microsoft.VisualBasic.FileIO.TextFieldParser]::new($Path, [System.Text.Encoding]::UTF8)
    $parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
    $parser.SetDelimiters(',')
    $parser.HasFieldsEnclosedInQuotes = $true
    try {
        if ($parser.EndOfData) { return @() }
        $headers = New-UniqueCsvHeader $parser.ReadFields()
        $rows = @()
        while (-not $parser.EndOfData) {
            $fields = $parser.ReadFields()
            $obj = [ordered]@{}
            for ($i = 0; $i -lt $headers.Count; ++$i) {
                $obj[$headers[$i]] = if ($i -lt $fields.Count) { $fields[$i] } else { '' }
            }
            $rows += [pscustomobject]$obj
        }
        return $rows
    } finally {
        $parser.Close()
    }
}

function Mean {
    param([object[]]$Items, [string]$PropertyName)
    $values = @()
    foreach ($item in $Items) {
        $value = PickProperty $item $PropertyName
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

function CsvSuccessToStatus {
    param([object]$Value)
    $text = ([string]$Value).Trim()
    if ($text -eq '1' -or $text -ieq 'true' -or $text -ieq 'ok') {
        return 'OK'
    }
    return 'FAILED'
}

function DetectCsvSchema {
    param([object]$Row, [string]$Family)
    if (HasColumn $Row 'num_structures_first' -or HasColumn $Row $StructureFirstColumn) { return 'structure' }
    if (HasColumn $Row 'num_correspondences' -or HasColumn $Row $DirectConfidenceColumn) { return 'direct' }
    if (HasColumn $Row 'num_learning_points_first' -or HasColumn $Row $LearningFirstColumn) { return 'learning' }
    if (-not [string]::IsNullOrWhiteSpace($Family)) { return $Family }
    return 'keypoint'
}

function SchemaPrimaryLabelHtml {
    param([string]$Schema)
    switch ($Schema) {
        'structure' { return '&#32467;&#26500;&#25968;' }
        'direct' { return '&#30452;&#25509;&#27861;&#32622;&#20449;&#24230;' }
        'learning' { return '&#23398;&#20064;&#28857;&#25968;' }
        default { return '&#20851;&#38190;&#28857;&#25968;' }
    }
}

function GetBatchPathParts {
    param([string]$RelativePath)
    $parts = $RelativePath -split '/'
    $batchIndex = [array]::IndexOf($parts, 'batch')
    if ($batchIndex -lt 0 -or $parts.Count -lt ($batchIndex + 4)) {
        return [pscustomobject]@{ Family='unknown'; Pipeline='unknown' }
    }
    return [pscustomobject]@{
        Family = $parts[$batchIndex + 1]
        Pipeline = $parts[$batchIndex + 2]
    }
}

function New-ReportRecordFromCsvRow {
    param([object]$Row, [string]$CsvPath)
    $relativeCsv = RelPath $CsvPath
    $pathParts = GetBatchPathParts $relativeCsv
    $family = $pathParts.Family
    $pipeline = $pathParts.Pipeline
    $schema = DetectCsvSchema $Row $family
    $sample = PickFirst $Row @('sample_name', $SampleNameColumn, 'sample')

    $countFirst = PickFirst $Row @(
        'num_keypoints_first', $KeypointFirstColumn,
        'num_structures_first', $StructureFirstColumn,
        'num_learning_points_first', $LearningFirstColumn
    )
    $countSecond = PickFirst $Row @(
        'num_keypoints_second', $KeypointSecondColumn,
        'num_structures_second', $StructureSecondColumn,
        'num_learning_points_second', $LearningSecondColumn
    )
    $primaryCount = if ($schema -eq 'direct') {
        PickFirst $Row @($DirectConfidenceColumn, 'direct_confidence', 'num_correspondences', 'num_raw_matches')
    } elseif ($null -ne $countFirst -or $null -ne $countSecond) {
        '{0} / {1}' -f (FormatNumber $countFirst 0), (FormatNumber $countSecond 0)
    } else {
        '-'
    }

    [pscustomobject]@{
        Family = $family
        Pipeline = $pipeline
        Sample = $sample
        Status = CsvSuccessToStatus (PickFirst $Row @('success', $SuccessColumn))
        Message = PickFirst $Row @('message', $MessageColumn)
        SummaryPath = $relativeCsv
        Schema = $schema
        PrimaryLabelHtml = SchemaPrimaryLabelHtml $schema
        PrimaryCount = $primaryCount
        RawMatches = if ($schema -eq 'direct') { $null } else { PickFirst $Row @(
            'num_candidate_structure_matches',
            'num_raw_learning_matches',
            'num_correspondences',
            'num_raw_matches'
        ) }
        FilteredMatches = if ($schema -eq 'direct') { $null } else { PickFirst $Row @(
            'num_filtered_structure_matches',
            'num_filtered_learning_matches',
            'num_filtered_matches',
            'num_correspondences'
        ) }
        Inliers = PickFirst $Row @(
            'num_inlier_structure_matches',
            'num_inlier_learning_matches',
            'num_inlier_correspondences',
            'num_inliers',
            $InitializerInliersColumn
        )
        InlierRatio = PickFirst $Row @(
            'structure_inlier_ratio',
            'learning_inlier_ratio',
            'direct_confidence',
            'inlier_ratio',
            $DirectConfidenceColumn,
            $InitializerInlierRatioColumn
        )
        ReprojError = PickFirst $Row @('mean_structure_reproj_error', 'mean_reproj_error')
        IoU = PickFirst $Row @('warp_overlap_iou', $EdgeAlignmentColumn)
        TotalMs = PickFirst $Row @('t_total_ms', $TotalMsColumn)
    }
}

function Read-SummaryCsvRecords {
    $records = @()
    $summaryCsvFiles = @(Get-ChildItem -Path $BatchRoot -Recurse -Filter summary.csv | Sort-Object FullName)
    foreach ($file in $summaryCsvFiles) {
        $rows = @(Read-CsvRowsUnique $file.FullName)
        foreach ($row in $rows) {
            $records += New-ReportRecordFromCsvRow $row $file.FullName
        }
    }
    return $records
}

function New-ReportRecordFromJson {
    param([System.IO.FileInfo]$File)
    $json = Get-Content -LiteralPath $File.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $relativeSummary = RelPath $File.FullName
    $parts = $relativeSummary -split '/'
    $counts = PickProperty $json 'counts'
    $quality = PickProperty $json 'quality'
    $timings = PickProperty $json 'timings_ms'
    $family = if (PickProperty $json 'method_family') { PickProperty $json 'method_family' } else { $parts[2] }
    $schema = DetectCsvSchema $counts $family
    $countFirst = PickFirst $counts @('num_keypoints_first', 'num_structures_first')
    $countSecond = PickFirst $counts @('num_keypoints_second', 'num_structures_second')
    $primaryCount = if ($schema -eq 'direct') {
        PickFirst $counts @('direct_confidence', 'num_correspondences', 'num_raw_matches')
    } else {
        '{0} / {1}' -f (FormatNumber $countFirst 0), (FormatNumber $countSecond 0)
    }

    [pscustomobject]@{
        Family = $family
        Pipeline = if (PickProperty $json 'pipeline_name') { PickProperty $json 'pipeline_name' } else { $parts[3] }
        Sample = if (PickProperty $json 'sample_name') { PickProperty $json 'sample_name' } else { $parts[4] }
        Status = PickProperty $json 'status'
        Message = PickProperty $json 'message'
        SummaryPath = $relativeSummary
        Schema = $schema
        PrimaryLabelHtml = SchemaPrimaryLabelHtml $schema
        PrimaryCount = $primaryCount
        RawMatches = if ($schema -eq 'direct') { $null } else { PickFirst $counts @('num_raw_matches', 'num_correspondences') }
        FilteredMatches = if ($schema -eq 'direct') { $null } else { PickFirst $counts @('num_filtered_matches', 'num_correspondences') }
        Inliers = PickFirst $counts @('num_inliers', 'feature_initializer_inliers')
        InlierRatio = PickFirst $quality @('inlier_ratio', 'direct_confidence', 'feature_initializer_inlier_ratio')
        ReprojError = PickProperty $quality 'mean_reproj_error'
        IoU = PickFirst $quality @('warp_overlap_iou', 'warp_edge_alignment_iou')
        TotalMs = PickProperty $timings 'total'
    }
}

function Read-FallbackJsonRecords {
    $summaryFiles = @(Get-ChildItem -Path $BatchRoot -Recurse -Filter run_summary.json | Sort-Object FullName)
    return @($summaryFiles | ForEach-Object { New-ReportRecordFromJson $_ })
}

if (-not (Test-Path $BatchRoot)) {
    throw "Batch output directory not found: $BatchRoot"
}

$records = @(Read-SummaryCsvRecords)
if ($records.Count -eq 0) {
    $records = @(Read-FallbackJsonRecords)
}

$samples = Get-ChildItem -Path $DatasetRoot -Directory | Sort-Object Name | ForEach-Object {
    $source = Get-ChildItem -Path $_.FullName -File | Where-Object { $_.BaseName -match 'source|moving' } | Select-Object -First 1
    $target = Get-ChildItem -Path $_.FullName -File | Where-Object { $_.BaseName -match 'target|reference' } | Select-Object -First 1
    [pscustomobject]@{
        Name = $_.Name
        Source = if ($source) { RelPath $source.FullName } else { '' }
        Target = if ($target) { RelPath $target.FullName } else { '' }
        SourceSizeKB = if ($source) { [math]::Round($source.Length / 1KB, 1) } else { $null }
        TargetSizeKB = if ($target) { [math]::Round($target.Length / 1KB, 1) } else { $null }
    }
}

$totalCases = $records.Count
$okCases = @($records | Where-Object { $_.Status -eq 'OK' }).Count
$failedCases = $totalCases - $okCases
$families = @($records | Select-Object -ExpandProperty Family -Unique | Sort-Object)
$pipelines = @($records | Select-Object -ExpandProperty Pipeline -Unique | Sort-Object)
$sampleNames = @($samples | Select-Object -ExpandProperty Name)
$avgMs = Mean $records 'TotalMs'
$generatedAt = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')

$configRows = @(
    [pscustomobject]@{NameHtml='&#28857;&#29305;&#24449;&#27861;&#25209;&#37327;&#27979;&#35797;'; Path='configs/pipeline/batch/batch_keypoint.yaml'; Command='build-mingw\registration_app.exe configs\pipeline\batch\batch_keypoint.yaml'},
    [pscustomobject]@{NameHtml='&#32467;&#26500;&#29305;&#24449;&#27861;&#25209;&#37327;&#27979;&#35797;'; Path='configs/pipeline/batch/batch_structure.yaml'; Command='build-mingw\registration_app.exe configs\pipeline\batch\batch_structure.yaml'},
    [pscustomobject]@{NameHtml='&#30452;&#25509;&#27861;&#25209;&#37327;&#27979;&#35797;'; Path='configs/pipeline/batch/batch_direct.yaml'; Command='build-mingw\registration_app.exe configs\pipeline\batch\batch_direct.yaml'},
    [pscustomobject]@{NameHtml='&#28145;&#24230;&#23398;&#20064;&#21305;&#37197;&#25209;&#37327;&#27979;&#35797;'; Path='configs/pipeline/batch/batch_learning.yaml'; Command='build-mingw\registration_app.exe configs\pipeline\batch\batch_learning.yaml'}
) | ForEach-Object {
    '<tr><td>{0}</td><td>{1}</td><td>{2}</td></tr>' -f $_.NameHtml, (HtmlEscape $_.Path), (HtmlEscape $_.Command)
}

$sampleRows = foreach ($sample in $samples) {
    $caseCount = @($records | Where-Object { $_.Sample -eq $sample.Name }).Count
    '<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3} KB</td><td>{4} KB</td><td>{5}</td></tr>' -f `
        (HtmlEscape $sample.Name), (HtmlEscape $sample.Source), (HtmlEscape $sample.Target), (FormatNumber $sample.SourceSizeKB 1), (FormatNumber $sample.TargetSizeKB 1), $caseCount
}

$pipelineRows = foreach ($group in ($records | Group-Object Family, Pipeline | Sort-Object Name)) {
    $items = @($group.Group)
    $first = $items[0]
    $ok = @($items | Where-Object { $_.Status -eq 'OK' }).Count
    '<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3} / {2}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td></tr>' -f `
        (HtmlEscape $first.Family),
        (HtmlEscape $first.Pipeline),
        $items.Count,
        $ok,
        (FormatNumber (Mean $items 'IoU')),
        (FormatNumber (Mean $items 'InlierRatio')),
        (FormatNumber (Mean $items 'TotalMs') 1),
        (HtmlEscape $first.SummaryPath)
}

$caseRows = foreach ($record in $records) {
    $statusText = if ($record.Status -eq 'OK') { '&#36890;&#36807;' } else { '&#26410;&#36890;&#36807;' }
    '<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}<br><span class="muted">{6}</span></td><td>{7}</td><td>{8}</td><td>{9}</td><td>{10}</td><td>{11}</td><td>{12}</td><td>{13}</td><td>{14}</td></tr>' -f `
        (HtmlEscape $record.Sample),
        (HtmlEscape $record.Family),
        (HtmlEscape $record.Pipeline),
        $statusText,
        (HtmlEscape $record.Message),
        $record.PrimaryLabelHtml,
        (HtmlEscape $record.PrimaryCount),
        (FormatNumber $record.RawMatches 0),
        (FormatNumber $record.FilteredMatches 0),
        (FormatNumber $record.Inliers 0),
        (FormatNumber $record.InlierRatio),
        (FormatNumber $record.ReprojError),
        (FormatNumber $record.IoU),
        (FormatNumber $record.TotalMs 1),
        (HtmlEscape $record.SummaryPath)
}

$template = @'
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>&#22270;&#20687;&#37197;&#20934;&#23454;&#39564;&#24179;&#21488;&#27979;&#35797;&#29992;&#20363;&#25253;&#21578;</title>
  <style>
    body { font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; color: #111827; line-height: 1.45; }
    h1 { font-size: 24px; margin: 0 0 12px; }
    h2 { font-size: 18px; margin: 24px 0 10px; }
    p { margin: 0 0 12px; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0 18px; table-layout: auto; }
    th, td { border: 1px solid #cbd5e1; padding: 6px 8px; font-size: 12px; vertical-align: top; }
    th { background: #f1f5f9; font-weight: 700; }
    .muted { color: #64748b; }
  </style>
</head>
<body>
  <h1>&#22270;&#20687;&#37197;&#20934;&#23454;&#39564;&#24179;&#21488;&#27979;&#35797;&#29992;&#20363;&#25253;&#21578;</h1>
  <p>&#26412;&#25253;&#21578;&#20248;&#20808;&#26681;&#25454; outputs/batch &#19979;&#21508; pipeline &#30340; summary.csv &#33258;&#21160;&#25972;&#29702;&#65307;&#19981;&#21516;&#26041;&#27861;&#26063;&#20351;&#29992;&#21508;&#33258;&#30340; CSV schema&#65292;&#26087;&#36755;&#20986;&#32570;&#23569; summary.csv &#26102;&#25165;&#22238;&#36864;&#35835;&#21462; run_summary.json&#12290;</p>
  <h2>1. &#24635;&#20307;&#32479;&#35745;</h2>
  <table>
    <tr><th>&#25351;&#26631;</th><th>&#25968;&#20540;</th></tr>
    <tr><td>&#27979;&#35797;&#29992;&#20363;&#24635;&#25968;</td><td>__TOTAL_CASES__</td></tr>
    <tr><td>&#36890;&#36807;&#29992;&#20363;</td><td>__OK_CASES__</td></tr>
    <tr><td>&#26410;&#36890;&#36807;&#29992;&#20363;</td><td>__FAILED_CASES__</td></tr>
    <tr><td>Pipeline &#25968;&#37327;</td><td>__PIPELINE_COUNT__</td></tr>
    <tr><td>&#26041;&#27861;&#26063;&#25968;&#37327;</td><td>__FAMILY_COUNT__</td></tr>
    <tr><td>&#26679;&#26412;&#25968;&#37327;</td><td>__SAMPLE_COUNT__</td></tr>
    <tr><td>&#24179;&#22343;&#32791;&#26102;</td><td>__AVG_MS__ ms</td></tr>
  </table>
  <h2>2. &#25209;&#37327;&#27979;&#35797;&#20837;&#21475;</h2>
  <table>
    <tr><th>&#27979;&#35797;&#31867;&#22411;</th><th>&#37197;&#32622;&#25991;&#20214;</th><th>&#36816;&#34892;&#21629;&#20196;</th></tr>
    __CONFIG_ROWS__
  </table>
  <h2>3. &#25968;&#25454;&#38598;&#26679;&#26412;</h2>
  <table>
    <tr><th>&#26679;&#26412;</th><th>&#28304;&#22270;</th><th>&#30446;&#26631;&#22270;</th><th>&#28304;&#22270;&#22823;&#23567;</th><th>&#30446;&#26631;&#22270;&#22823;&#23567;</th><th>&#24050;&#25191;&#34892;&#29992;&#20363;&#25968;</th></tr>
    __SAMPLE_ROWS__
  </table>
  <h2>4. Pipeline &#27719;&#24635;</h2>
  <table>
    <tr><th>&#26041;&#27861;&#26063;</th><th>Pipeline</th><th>&#29992;&#20363;&#25968;</th><th>&#36890;&#36807;&#25968;</th><th>&#24179;&#22343; IoU</th><th>&#24179;&#22343;&#20869;&#28857;/&#32622;&#20449;&#24230;</th><th>&#24179;&#22343;&#32791;&#26102; ms</th><th>CSV &#27719;&#24635;</th></tr>
    __PIPELINE_ROWS__
  </table>
  <h2>5. &#27979;&#35797;&#29992;&#20363;&#26126;&#32454;</h2>
  <table>
    <tr><th>&#26679;&#26412;</th><th>&#26041;&#27861;&#26063;</th><th>Pipeline</th><th>&#29366;&#24577;</th><th>&#35828;&#26126;</th><th>&#23545;&#35937;&#25968;&#37327;</th><th>&#20505;&#36873;/&#21407;&#22987;&#21305;&#37197;</th><th>&#36807;&#28388;&#21305;&#37197;</th><th>&#20869;&#28857;</th><th>&#20869;&#28857;&#29575;/&#32622;&#20449;&#24230;</th><th>&#37325;&#25237;&#24433;&#35823;&#24046;</th><th>IoU</th><th>&#32791;&#26102; ms</th><th>&#20449;&#24687;&#26469;&#28304;</th></tr>
    __CASE_ROWS__
  </table>
  <p>&#29983;&#25104;&#26102;&#38388;&#65306;__GENERATED_AT__</p>
</body>
</html>
'@

$html = $template
$replacements = @{
    '__TOTAL_CASES__' = [string]$totalCases
    '__OK_CASES__' = [string]$okCases
    '__FAILED_CASES__' = [string]$failedCases
    '__PIPELINE_COUNT__' = [string]$pipelines.Count
    '__FAMILY_COUNT__' = [string]$families.Count
    '__SAMPLE_COUNT__' = [string]$sampleNames.Count
    '__AVG_MS__' = (FormatNumber $avgMs 1)
    '__CONFIG_ROWS__' = ($configRows -join "`n")
    '__SAMPLE_ROWS__' = ($sampleRows -join "`n")
    '__PIPELINE_ROWS__' = ($pipelineRows -join "`n")
    '__CASE_ROWS__' = ($caseRows -join "`n")
    '__GENERATED_AT__' = (HtmlEscape $generatedAt)
}

foreach ($key in $replacements.Keys) {
    $html = $html.Replace($key, $replacements[$key])
}

$encoding = [System.Text.UTF8Encoding]::new($false)
[System.IO.Directory]::CreateDirectory((Split-Path -Path $OutputFile -Parent)) | Out-Null
[System.IO.File]::WriteAllText($OutputFile, $html, $encoding)

Write-Output "Generated: $OutputFile"
