$repo = "C:\Ai_Expert\L53-Homework"
$dl   = "$HOME\Downloads"

cd $repo
New-Item -ItemType Directory -Force -Path "$repo\tests" | Out-Null

# מפה: pattern לחיפוש -> יעד סופי
$mapping = @(
    @{ Pattern = "README*.md";              Dest = "$repo\README.md"            }
    @{ Pattern = "readme_md_file.md";       Dest = "$repo\README.md"            }   # fallback
    @{ Pattern = "CHANGELOG*.md";           Dest = "$repo\CHANGELOG.md"         }
    @{ Pattern = "evaluation_graph*.png";   Dest = "$repo\evaluation_graph.png" }
    @{ Pattern = "risk_metrics*.py";        Dest = "$repo\src\risk_metrics.py"  }
    @{ Pattern = "significance_tests*.py";  Dest = "$repo\src\significance_tests.py" }
    @{ Pattern = "test_no_leakage*.py";     Dest = "$repo\tests\test_no_leakage.py"  }
    @{ Pattern = "test_metrics*.py";        Dest = "$repo\tests\test_metrics.py"     }
    @{ Pattern = "test_gatekeeper*.py";     Dest = "$repo\tests\test_gatekeeper.py"  }
)

$copiedDest = @{}   # למניעת כפילות (README יש שני patterns)

foreach ($entry in $mapping) {
    if ($copiedDest.ContainsKey($entry.Dest)) { continue }

    # חיפוש רק בשורש Downloads (Depth 0) — מתעלם מ-Market-Basket-Analysis
    $candidates = Get-ChildItem -Path $dl -Filter $entry.Pattern -File -ErrorAction SilentlyContinue |
                  Sort-Object LastWriteTime -Descending

    if ($candidates) {
        $chosen = $candidates[0]
        Copy-Item $chosen.FullName $entry.Dest -Force
        Write-Host ("OK   {0,-30}  <-  {1}  [{2:yyyy-MM-dd HH:mm}]" -f `
                    (Split-Path $entry.Dest -Leaf), $chosen.Name, $chosen.LastWriteTime) -ForegroundColor Green
        if ($candidates.Count -gt 1) {
            Write-Host ("     ({0} candidates: {1})" -f `
                        $candidates.Count, ($candidates.Name -join ", ")) -ForegroundColor DarkGray
        }
        $copiedDest[$entry.Dest] = $true
    } else {
        Write-Host ("MISS {0}  (no match for pattern {1})" -f `
                    (Split-Path $entry.Dest -Leaf), $entry.Pattern) -ForegroundColor Yellow
    }
}

# tests\__init__.py - קובץ ריק
New-Item -ItemType File -Force -Path "$repo\tests\__init__.py" | Out-Null
Write-Host "OK   tests\__init__.py created (empty)" -ForegroundColor Green

# סיכום
Write-Host "`n=== Final state in repo ===" -ForegroundColor Cyan
Get-ChildItem "$repo\README.md","$repo\CHANGELOG.md","$repo\evaluation_graph.png" -ErrorAction SilentlyContinue |
    Format-Table Name,Length,LastWriteTime
Get-ChildItem "$repo\src\risk_metrics.py","$repo\src\significance_tests.py" -ErrorAction SilentlyContinue |
    Format-Table Name,Length,LastWriteTime
Get-ChildItem "$repo\tests\*.py" -ErrorAction SilentlyContinue |
    Format-Table Name,Length,LastWriteTime