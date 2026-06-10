# Re-process every case in _phi/ WITHOUT re-running Textract.
#
# Usage (from the repo root, in PowerShell):
#   .\batch_rerun.ps1                    # all cases under _phi/
#   .\batch_rerun.ps1 -CaseIds 2010125,2218548  # just those
#   .\batch_rerun.ps1 -SkipMatcher       # re-chunk + re-load only (no matcher)
#   .\batch_rerun.ps1 -SkipRechunk       # re-load + re-run matcher only
#
# Three stages per case (each optional via flags):
#   1. rechunk_from_raw     — re-chunk the saved Textract JSON with the
#                             latest chunker + allegation extractor (no $)
#   2. load_case_to_db      — re-embed chunks + allegations, refresh DB
#                             (~$0.05 in Bedrock embeddings per case)
#   3. run.py --from-db     — re-run matcher (~$2-6 in Bedrock LLM)
#
# Resumability: the per-case loop catches non-zero exit codes and
# continues. Failures are summarized at the end.
#
# Cost: ~$2-6 per case (skip-Textract savings vs ~$5-7 for full ingest).
# Time: ~3-5 min per case sequentially.

param(
    [string[]]$CaseIds = @(),    # explicit list; if empty, discover from _phi/
    [switch]$SkipRechunk,        # skip Stage 1 (use existing chunks.json)
    [switch]$SkipLoad,           # skip Stage 2 (DB already current)
    [switch]$SkipMatcher,        # skip Stage 3 (no matcher re-run)
    [switch]$Force               # re-run even if output/<case>/0_CASE_SUMMARY.html exists
)

$logFile = "batch_rerun_$(Get-Date -Format 'yyyyMMdd-HHmmss').log"

# Sanity checks
if (-not $env:DATABASE_URL)   { Write-Error "DATABASE_URL not set";   exit 1 }
if (-not $env:PGPASSWORD)     { Write-Error "PGPASSWORD not set";     exit 1 }
if (-not $env:OUTPUT_PUBLISH_BUCKET) {
    Write-Warning "OUTPUT_PUBLISH_BUCKET not set; outputs will NOT be auto-published to S3."
}

# AWS SSO pre-flight (Stage 2 + Stage 3 both call Bedrock)
if (-not ($SkipLoad -and $SkipMatcher)) {
    Write-Host "Verifying AWS SSO token..."
    aws sts get-caller-identity --profile user --output text > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "AWS SSO token expired. Run: aws sso login --profile user"
        exit 2
    }
    Write-Host "  SSO token OK"
}

# Discover case IDs from _phi/ if not provided
if ($CaseIds.Count -eq 0) {
    if (-not (Test-Path "_phi")) {
        Write-Error "_phi/ doesn't exist; pass case IDs explicitly via -CaseIds"
        exit 3
    }
    $CaseIds = Get-ChildItem "_phi" -Directory |
               Where-Object { Test-Path (Join-Path $_.FullName "textract_raw.json") } |
               ForEach-Object { $_.Name } |
               Sort-Object
}

if ($CaseIds.Count -eq 0) {
    Write-Error "No cases with cached Textract found under _phi/"
    exit 4
}

Write-Host ""
Write-Host "Re-processing $($CaseIds.Count) case(s): $($CaseIds -join ', ')"
Write-Host "  log -> $logFile"

$start    = Get-Date
$ok       = @()
$failed   = @()
$skipped  = @()

foreach ($case_id in $CaseIds) {
    $sep = "=" * 75

    # Resumability check
    $marker = Join-Path "output" $case_id | Join-Path -ChildPath "0_CASE_SUMMARY.html"
    if ((Test-Path $marker) -and (-not $Force)) {
        Write-Host ""
        Write-Host "[$($CaseIds.IndexOf($case_id) + 1)/$($CaseIds.Count)] SKIP $case_id  (already has $marker; pass -Force to re-run)"
        $skipped += $case_id
        continue
    }

    Write-Host ""
    Write-Host $sep
    Write-Host "[$($CaseIds.IndexOf($case_id) + 1)/$($CaseIds.Count)] RE-PROCESSING $case_id"
    Write-Host $sep

    $caseFailed = $false

    # Stage 1: rechunk from saved Textract
    if (-not $SkipRechunk) {
        Write-Host "  [1/3] rechunk_from_raw('$case_id')..."
        py -c "from pipeline.ingest_real import rechunk_from_raw; rechunk_from_raw('$case_id')" 2>&1 |
            Tee-Object -FilePath $logFile -Append
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    FAILED rechunk for $case_id"
            $caseFailed = $true
            $failed += "$case_id (rechunk)"
            continue
        }
    }

    # Stage 2: re-load to DB with fresh embeddings
    if (-not $SkipLoad) {
        Write-Host "  [2/3] load_case_to_db.py $case_id..."
        py db\load_case_to_db.py $case_id 2>&1 |
            Tee-Object -FilePath $logFile -Append
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    FAILED load for $case_id"
            $caseFailed = $true
            $failed += "$case_id (load)"
            continue
        }
    }

    # Stage 3: re-run matcher
    if (-not $SkipMatcher) {
        Write-Host "  [3/3] run.py --from-db $case_id..."
        py run.py --from-db $case_id 2>&1 |
            Tee-Object -FilePath $logFile -Append
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    FAILED matcher for $case_id"
            $caseFailed = $true
            $failed += "$case_id (matcher)"
            # SSO might have expired — fast-fail the loop if so
            aws sts get-caller-identity --profile user --output text > $null 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Error "AWS SSO token expired mid-batch. Stopping."
                break
            }
            continue
        }
    }

    if (-not $caseFailed) {
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
Write-Host "Cases attempted:  $($CaseIds.Count)"
Write-Host "Skipped ($($skipped.Count)):     $($skipped -join ', ')"
Write-Host "OK      ($($ok.Count)):          $($ok -join ', ')"
Write-Host "Failed  ($($failed.Count)):     $($failed -join ', ')"
Write-Host ""
Write-Host "Full log: $logFile"
