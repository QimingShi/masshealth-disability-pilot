# Batch-run the full ingest+matcher pipeline on a list of case folders.
#
# Usage (from the repo root):
#   .\batch_ingest.ps1
#
# Edit $folders below to add/remove cases. Each entry is the S3 prefix
# under the incoming bucket (case_id is auto-extracted from the leading
# digits — works for any digit length).
#
# Prereqs (once per shell):
#   aws sso login --profile user
#   $env:DATABASE_URL = "postgresql://<user>@<host>:5432/<db>"
#   $env:PGPASSWORD   = "<your password>"
#   $env:OUTPUT_PUBLISH_BUCKET = "umasschan-forhealth-expedite-outgoing-data-nonprod"
#
# Cost: ~$5-7 per case (Textract is the dominant cost). For 16 cases ≈ $80-110.
# Time: ~5-10 min per case sequentially → ~1.5-2.5 hours total.

$bucket  = "umasschan-forhealth-expedite-incoming-data-nonprod"
$logFile = "batch_ingest_$(Get-Date -Format 'yyyyMMdd-HHmmss').log"

$folders = @(
    "124264 Med - 120/",
    "2010125 Med - 100/",
    "2011162 Psych-120/",
    "2173204 Med - 100/",
    "2181878 Psych- 100/",
    "2182555 Psych- 100/",
    "2198279- 231/",
    "2199880 Psych- 120/",
    "2203790 Med - 120/",
    "2204762 - Med 120/",
    "2208661 Med - 120/",
    "2209253 Psych- 120/",
    "2218548 Psych- 120/",
    "2220672 -230/",
    "2231212- 210/",
    "7777777 Redacted/"
)

# Sanity checks before spending money
if (-not $env:DATABASE_URL)   { Write-Error "DATABASE_URL not set";   exit 1 }
if (-not $env:PGPASSWORD)     { Write-Error "PGPASSWORD not set";     exit 1 }
if (-not $env:OUTPUT_PUBLISH_BUCKET) {
    Write-Warning "OUTPUT_PUBLISH_BUCKET not set; outputs will NOT be auto-published to S3."
}

$start    = Get-Date
$ok       = @()
$failed   = @()

foreach ($folder in $folders) {
    # Extract leading digits as case_id (handles 6/7/8-digit MassHealth ids)
    if ($folder -match "^(\d+)") {
        $case_id = $Matches[1]
    } else {
        Write-Host "SKIP: couldn't parse case_id from '$folder'"
        $failed += $folder
        continue
    }

    $sep = "=" * 75
    Write-Host ""
    Write-Host $sep
    Write-Host "[$($folders.IndexOf($folder) + 1)/$($folders.Count)] INGESTING $case_id  (from '$folder')"
    Write-Host $sep
    Write-Host "  log -> $logFile"

    # Tee the per-case output to the log file
    py db\ingest_s3_to_db.py $case_id --folder "$bucket/$folder" 2>&1 | Tee-Object -FilePath $logFile -Append

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED $case_id (exit $LASTEXITCODE)"
        $failed += "$case_id ($folder)"
    } else {
        Write-Host "  OK $case_id"
        $ok += $case_id
    }
}

$elapsed = (Get-Date) - $start
$sep = "=" * 75
Write-Host ""
Write-Host $sep
Write-Host "BATCH COMPLETE in $($elapsed.TotalMinutes.ToString('F1')) minutes"
Write-Host $sep
Write-Host "Total cases attempted: $($folders.Count)"
Write-Host "OK     ($($ok.Count)):     $($ok -join ', ')"
Write-Host "Failed ($($failed.Count)): $($failed -join ', ')"
Write-Host ""
Write-Host "Full log: $logFile"
