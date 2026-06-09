# Batch-run the full ingest+matcher pipeline on a list of case folders.
#
# Usage (from the repo root, in PowerShell):
#   .\batch_ingest.ps1            # skip cases that already completed
#   .\batch_ingest.ps1 -Force     # re-run everything regardless
#
# Resumable: a case is considered "done" if output/<case_id>/0_CASE_SUMMARY.html
# exists. After an SSO expiry / network blip, just re-login and re-run --
# completed cases get skipped, only the rest re-ingest.
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

param(
    [switch]$Force   # re-run cases even if output/<case>/0_CASE_SUMMARY.html exists
)

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

# Pre-flight: verify AWS SSO token is still valid. SSO tokens expire after
# ~8 hours and the failure mid-batch is silent (boto3 raises 60-90 sec into
# the Textract call). Fail fast with a clear message before doing anything.
Write-Host "Verifying AWS SSO token..."
aws sts get-caller-identity --profile user --output text > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "AWS SSO token expired or invalid. Run: aws sso login --profile user"
    exit 2
}
Write-Host "  SSO token OK"

$start    = Get-Date
$ok       = @()
$failed   = @()
$skipped  = @()

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

    # Resumability: skip if the case already has a rendered case-summary
    # HTML. Lets you re-run the batch after an SSO expiry / network blip
    # without paying for already-completed Textract jobs again. Override
    # by passing -Force or by deleting output/<case_id>/.
    $marker = Join-Path "output" $case_id | Join-Path -ChildPath "0_CASE_SUMMARY.html"
    if ((Test-Path $marker) -and (-not $Force)) {
        Write-Host ""
        Write-Host "[$($folders.IndexOf($folder) + 1)/$($folders.Count)] SKIP $case_id  (already has $marker; pass -Force to re-run)"
        $skipped += $case_id
        continue
    }

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
        # Common cause: SSO token expired mid-batch. Re-check and short-
        # circuit so we don't churn through the rest of the list with
        # guaranteed failures.
        aws sts get-caller-identity --profile user --output text > $null 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "AWS SSO token expired mid-batch. Stopping. Run: aws sso login --profile user"
            break
        }
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
Write-Host "Total cases in list:   $($folders.Count)"
Write-Host "Skipped ($($skipped.Count)) — already done: $($skipped -join ', ')"
Write-Host "OK      ($($ok.Count)):                     $($ok -join ', ')"
Write-Host "Failed  ($($failed.Count)):                     $($failed -join ', ')"
Write-Host ""
Write-Host "Full log: $logFile"
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "To retry just the failures: re-run .\batch_ingest.ps1 -- failed"
    Write-Host "cases will be retried automatically (they don't have a 0_CASE_SUMMARY.html)."
}
