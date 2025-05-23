# PowerShell script to deploy the Google Cloud Function

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

# Check if required environment variables are set
$requiredVars = @("PROJECT_ID", "REGION", "FUNCTION_NAME")
$missingVars = @()

foreach ($var in $requiredVars) {
    if (-not (Get-Item "Env:$var" -ErrorAction SilentlyContinue)) {
        $missingVars += $var
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "Missing required environment variables: $($missingVars -join ', ')" -ForegroundColor Red
    Write-Host "Please copy .env.example to .env and update values" -ForegroundColor Red
    exit 1
}

# Deploy the function
Write-Host "Deploying function $env:FUNCTION_NAME to project $env:PROJECT_ID in region $env:REGION..." -ForegroundColor Cyan

$runtimeFlag = if ($env:RUNTIME) { "--runtime=$env:RUNTIME" } else { "--runtime=python310" }
$memoryFlag = if ($env:MEMORY) { "--memory=$env:MEMORY" } else { "--memory=1024MB" }
$timeoutFlag = if ($env:TIMEOUT) { "--timeout=$env:TIMEOUT" } else { "--timeout=300s" }

# First, deploy the health check function to verify everything is working
$healthCheckCmd = "gcloud functions deploy health-check " +
             "--entry-point=health_check " +
             "$runtimeFlag " +
             "--trigger-http " +
             "--allow-unauthenticated " +
             "--memory=256MB " +
             "--timeout=60s " +
             "--region=$env:REGION " +
             "--project=$env:PROJECT_ID"

Write-Host "Running command: $healthCheckCmd" -ForegroundColor Yellow
Invoke-Expression $healthCheckCmd

# If health check deployed successfully, deploy the main function
if ($LASTEXITCODE -eq 0) {
    Write-Host "Health check function deployed successfully, now deploying main function..." -ForegroundColor Green
    
    $deployCmd = "gcloud functions deploy $env:FUNCTION_NAME " +
                 "--entry-point=timeseries_analysis " +
                 "$runtimeFlag " +
                 "--trigger-http " +
                 "--allow-unauthenticated " +
                 "$memoryFlag " +
                 "$timeoutFlag " +
                 "--region=$env:REGION " +
                 "--project=$env:PROJECT_ID"

    Write-Host "Running command: $deployCmd" -ForegroundColor Yellow
    Invoke-Expression $deployCmd
} else {
    Write-Host "Health check function deployment failed, check logs for details" -ForegroundColor Red
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "Function deployed successfully!" -ForegroundColor Green
    
    # Get the URL of the deployed function
    $getUrlCmd = "gcloud functions describe $env:FUNCTION_NAME --region=$env:REGION --project=$env:PROJECT_ID --format='value(httpsTrigger.url)'"
    $functionUrl = Invoke-Expression $getUrlCmd
    
    Write-Host "Function URL: $functionUrl" -ForegroundColor Green
    Write-Host "You can test your function with: curl -X POST $functionUrl -H 'Content-Type: application/json' -d @test_data.json" -ForegroundColor Cyan
} else {
    Write-Host "Deployment failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
