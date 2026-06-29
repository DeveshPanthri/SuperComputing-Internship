
# Step 1: Extract Day1-Day6 files from git objects and write to new folder structure

$files = @{
    "SuperComputing_Day1_ML.ipynb"                              = "Day1\SuperComputing_Day1_ML.ipynb"
    "Day2_decision_tree_iris(1).ipynb"                          = "Day2\Day2_decision_tree_iris(1).ipynb"
    "Day2_linearreg.ipynb"                                      = "Day2\Day2_linearreg.ipynb"
    "Day2_svm-iris.ipynb"                                       = "Day2\Day2_svm-iris.ipynb"
    "Day3_supercomputing_unsupervised_clusteringipynb.ipynb"    = "Day3\Day3_supercomputing_unsupervised_clusteringipynb.ipynb"
    "Day4_Supercomputing_perceptron.ipynb"                      = "Day4\Day4_Supercomputing_perceptron.ipynb"
    "Day5_CNN_MNISTipynb (1).ipynb"                             = "Day5\Day5_CNN_MNISTipynb (1).ipynb"
    "Day6_RNN_LSTM_female_births.ipynb"                         = "Day6\Day6_RNN_LSTM_female_births.ipynb"
    "MACHINE LEARNING.docx"                                     = "MACHINE LEARNING.docx"
}

foreach ($src in $files.Keys) {
    $dst = $files[$src]
    $dstDir = Split-Path $dst -Parent
    if ($dstDir -and -not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        Write-Host "Created folder: $dstDir"
    }
    # Extract content from git and write to destination
    $bytes = git show "HEAD:$src" --format=raw 2>$null
    if ($LASTEXITCODE -eq 0) {
        git show "HEAD:$src" | Set-Content -Path $dst -Encoding Byte
        Write-Host "Extracted: $src -> $dst"
    } else {
        Write-Host "WARNING: Could not extract $src"
    }
}

# Step 2: Move Final_Project files/folders into Final_Project/ subfolder
$fpItems = @("app", "data", "notebooks", "reports", "src", "run_pipeline.ipynb", "requirements.txt")

if (-not (Test-Path "Final_Project")) {
    New-Item -ItemType Directory -Path "Final_Project" -Force | Out-Null
    Write-Host "Created folder: Final_Project"
}

foreach ($item in $fpItems) {
    if (Test-Path $item) {
        Move-Item -Path $item -Destination "Final_Project\$item" -Force
        Write-Host "Moved: $item -> Final_Project\$item"
    }
}

Write-Host "Done restructuring local files."
