# PowerShell script to deploy a simple test function to Google Cloud Functions

# Load environment variables from .env file if it exists
if (Test-Path .\.env) {
    Get-Content .\.env | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Item -Path "Env:$key" -Value $value
        }
    }
    Write-Host "Loaded environment variables from .env file"
} else {
    Write-Host "No .env file found, using defaults or existing environment variables"
}

# Deploy the test function
Write-Host "Deploying test function to project $env:PROJECT_ID in region $env:REGION..." -ForegroundColor Cyan

$deployCmd = "gcloud functions deploy test-function " +
             "--entry-point=health_check " +
             "--runtime=python310 " +
             "--trigger-http " +
             "--allow-unauthenticated " +
             "--memory=256MB " +
             "--timeout=60s " +
             "--region=$env:REGION " +
             "--project=$env:PROJECT_ID " +
             "--source=."

Write-Host "Running command: $deployCmd" -ForegroundColor Yellow
Invoke-Expression $deployCmd
