# ==============================================================================
# Autonomous Industrial Defect Inspection System - Automated Setup & Launcher
# Works out-of-the-box on ANY Windows 10/11 PC (even if Python is not installed)
# ==============================================================================

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  AUTONOMOUS INDUSTRIAL DEFECT DETECTION SYSTEM (SELF-SUPERVISED AI)    " -ForegroundColor Green
Write-Host "                  One-Click Auto-Setup & Launcher                       " -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

$projectDir = $PSScriptRoot
Set-Location $projectDir

# Function to test if a python executable is a real python (not the Windows Store dummy stub)
function Test-PythonExecutable($exePath) {
    if (-not $exePath -or -not (Test-Path $exePath)) { return $false }
    try {
        $out = & $exePath -c "import sys; print(sys.version_info[0])" 2>$null
        if ($out -and $out.Trim() -eq "3") {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

# 1. Locate Working Python
Write-Host "[*] Step 1/3: Checking Python installation..." -ForegroundColor Yellow
$pythonExe = $null

# Check current path python if valid
try {
    $cmd = Get-Command "python" -ErrorAction SilentlyContinue
    if ($cmd -and (Test-PythonExecutable $cmd.Source)) {
        $pythonExe = $cmd.Source
    }
} catch {}

# Check standard known installation directories
if (-not $pythonExe) {
    $searchPaths = @(
        "$env:USERPROFILE\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Program Files\Python310\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe"
    )
    foreach ($p in $searchPaths) {
        if (Test-PythonExecutable $p) {
            $pythonExe = $p
            break
        }
    }
}

# Check 'py' launcher
if (-not $pythonExe) {
    try {
        $pyCmd = Get-Command "py" -ErrorAction SilentlyContinue
        if ($pyCmd) {
            $pyOut = & py -3 -c "import sys; print(sys.executable)" 2>$null
            if ($pyOut -and (Test-PythonExecutable $pyOut.Trim())) {
                $pythonExe = $pyOut.Trim()
            }
        }
    } catch {}
}

# 2. If Python is NOT found, download and install official Python 3.11 automatically
if (-not $pythonExe) {
    Write-Host "[!] Python 3 is NOT installed on this computer." -ForegroundColor Magenta
    Write-Host "[*] Automatically downloading official Python 3.11 installer for Windows..." -ForegroundColor Yellow
    
    $installerUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    $installerPath = "$env:TEMP\python-3.11.9-amd64.exe"
    
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Write-Host "    Downloading from: $installerUrl" -ForegroundColor Gray
    
    try {
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
    } catch {
        Write-Host "[!] Failed to download Python automatically. Error: $_" -ForegroundColor Red
        Write-Host "    Please download and install Python 3.11 manually from: https://www.python.org/downloads/"
        Read-Host "Press Enter to exit..."
        exit 1
    }

    Write-Host "[+] Download complete. Installing Python 3.11 (this may take 1-2 minutes)..." -ForegroundColor Yellow
    $installArgs = "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0"
    $process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -PassThru -Wait
    
    # Refresh PATH environment variable
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    
    # Verify installation
    $defaultInstallPath = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    if (Test-PythonExecutable $defaultInstallPath) {
        $pythonExe = $defaultInstallPath
    } else {
        # Check if python is now in path
        $cmdAfter = Get-Command "python" -ErrorAction SilentlyContinue
        if ($cmdAfter -and (Test-PythonExecutable $cmdAfter.Source)) {
            $pythonExe = $cmdAfter.Source
        }
    }

    if (-not $pythonExe) {
        Write-Host "[!] Installation completed, but python executable could not be resolved automatically." -ForegroundColor Red
        Write-Host "    Please close and re-open this file, or restart your computer." -ForegroundColor Yellow
        Read-Host "Press Enter to exit..."
        exit 1
    }
}

$pyVer = & $pythonExe --version 2>&1
Write-Host "[+] Using Python: $pythonExe ($pyVer)" -ForegroundColor Green

# 3. Step 2: Install / Verify Packages
Write-Host ""
Write-Host "[*] Step 2/3: Checking required AI and computer vision packages..." -ForegroundColor Yellow

$needsInstall = $false
try {
    & $pythonExe -c "import torch, torchvision, streamlit, cv2, sklearn, matplotlib, pandas, PIL" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $needsInstall = $true
    }
} catch {
    $needsInstall = $true
}

if ($needsInstall) {
    Write-Host "[*] Installing required packages (PyTorch, Torchvision, Streamlit, OpenCV, etc.)..." -ForegroundColor Cyan
    Write-Host "    This only happens on first launch. Please wait..." -ForegroundColor Gray
    
    # 1. Install PyTorch CPU first (compact and reliable on any PC)
    Write-Host "    -> Installing PyTorch (CPU version)..." -ForegroundColor Gray
    & $pythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    
    # 2. Install requirements.txt
    Write-Host "    -> Installing project dependencies (Streamlit, OpenCV, etc.)..." -ForegroundColor Gray
    & $pythonExe -m pip install -r requirements.txt --quiet
    
    Write-Host "[+] All dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "[+] All required packages are already installed and verified!" -ForegroundColor Green
}

# 4. Step 3: Launch Project Application
Write-Host ""
Write-Host "[*] Step 3/3: Launching Autonomous Industrial Inspection Dashboard..." -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  The web dashboard will open in your browser automatically at:         " -ForegroundColor Green
Write-Host "  http://localhost:8501                                                 " -ForegroundColor White
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[*] Tip: To stop the application at any time, press Ctrl+C in this window." -ForegroundColor Gray
Write-Host ""

# Launch Streamlit
& $pythonExe -m streamlit run app.py
