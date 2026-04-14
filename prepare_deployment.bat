@echo off
REM Deployment Preparation Script for Windows
REM Run this before deploying to Streamlit Cloud

echo.
echo ========================================
echo   Nepal Real Estate App - Deployment Prep
echo ========================================
echo.

REM Step 1: Create .streamlit directory
echo [1/8] Creating .streamlit directory...
if not exist ".streamlit" mkdir .streamlit

REM Step 2: Create config.toml
echo [2/8] Creating Streamlit config...
(
echo [theme]
echo primaryColor = "#FF4B4B"
echo backgroundColor = "#FFFFFF"
echo secondaryBackgroundColor = "#F0F2F6"
echo textColor = "#262730"
echo font = "sans serif"
echo.
echo [server]
echo headless = true
echo port = 8501
echo enableCORS = false
echo enableXsrfProtection = false
echo.
echo [browser]
echo gatherUsageStats = false
) > .streamlit\config.toml
echo    Done!
echo.

REM Step 3: Check model files
echo [3/8] Checking model files...
set missing=0

if exist "xgboost_housing_final.pkl" (
    echo    [OK] xgboost_housing_final.pkl
) else (
    echo    [MISSING] xgboost_housing_final.pkl
    set missing=1
)

if exist "catboost_land_model_final.pkl" (
    echo    [OK] catboost_land_model_final.pkl
) else (
    echo    [MISSING] catboost_land_model_final.pkl
    set missing=1
)

if exist "catboost_lalpurja_house_v2_final.pkl" (
    echo    [OK] catboost_lalpurja_house_v2_final.pkl
) else (
    echo    [MISSING] catboost_lalpurja_house_v2_final.pkl
    set missing=1
)

if exist "catboost_lalpurja_model_final.pkl" (
    echo    [OK] catboost_lalpurja_model_final.pkl
) else (
    echo    [MISSING] catboost_lalpurja_model_final.pkl
    set missing=1
)

if exist "scaler_lalpurja_house_v2.pkl" (
    echo    [OK] scaler_lalpurja_house_v2.pkl
) else (
    echo    [MISSING] scaler_lalpurja_house_v2.pkl
    set missing=1
)

if %missing%==1 (
    echo.
    echo    WARNING: Some model files are missing!
)
echo.

REM Step 4: Check requirements.txt
echo [4/8] Checking requirements.txt...
if exist "requirements.txt" (
    echo    [OK] requirements.txt found
) else (
    echo    [MISSING] Creating requirements.txt...
    (
echo streamlit^>=1.32.0
echo pandas^>=2.0.0
echo numpy^>=2.1.0
echo plotly^>=5.18.0
echo scikit-learn^>=1.4.0
echo xgboost^>=2.0.0
echo catboost^>=1.2.0
echo joblib^>=1.3.0
echo langchain-core^>=0.1.0
echo langchain-community^>=0.0.20
echo langchain-openai^>=0.1.0
echo langchain-huggingface^>=0.0.3
echo faiss-cpu^>=1.7.4
echo sentence-transformers^>=2.2.2
echo openai^>=1.0.0
echo python-dotenv^>=1.0.0
    ) > requirements.txt
    echo    [OK] Created requirements.txt
)
echo.

REM Step 5: Check .gitignore
echo [5/8] Checking .gitignore...
if exist ".gitignore" (
    findstr /C:".env" .gitignore >nul
    if errorlevel 1 (
        echo    [WARNING] Adding .env to .gitignore...
        echo .env >> .gitignore
    ) else (
        echo    [OK] .env is in .gitignore
    )
    
    findstr /C:"venv/" .gitignore >nul
    if errorlevel 1 (
        echo    [WARNING] Adding venv/ to .gitignore...
        echo venv/ >> .gitignore
    ) else (
        echo    [OK] venv/ is in .gitignore
    )
) else (
    echo    [MISSING] Creating .gitignore...
    (
echo .env
echo venv/
echo __pycache__/
echo *.pyc
echo .DS_Store
echo .ipynb_checkpoints/
echo *.log
    ) > .gitignore
    echo    [OK] Created .gitignore
)
echo.

REM Step 6: Check .env
echo [6/8] Checking secrets...
if exist ".env" (
    echo    [OK] .env file found ^(for local development^)
    echo    [WARNING] Make sure .env is in .gitignore!
    echo    You'll need to add secrets manually in Streamlit Cloud
) else (
    echo    [INFO] .env file not found
    echo    You'll need to add secrets in Streamlit Cloud dashboard
)
echo.

REM Step 7: Check app_final.py
echo [7/8] Checking main app file...
if exist "app_final.py" (
    echo    [OK] app_final.py found
) else (
    echo    [ERROR] app_final.py NOT FOUND!
    echo    This is your main application file!
)
echo.

REM Step 8: Check Git
echo [8/8] Checking Git status...
if exist ".git" (
    echo    [OK] Git repository initialized
    git status --short >nul 2>&1
    if errorlevel 1 (
        echo    [INFO] Git is initialized
    ) else (
        echo    [INFO] Check git status for uncommitted changes
    )
) else (
    echo    [WARNING] Git not initialized
    echo    Run: git init
)
echo.

REM Summary
echo.
echo ========================================
echo   DEPLOYMENT READINESS SUMMARY
echo ========================================
echo.

set ready=1

if %missing%==0 (
    echo [OK] All model files present
) else (
    echo [ERROR] Missing model files
    set ready=0
)

if exist "requirements.txt" (
    echo [OK] requirements.txt ready
) else (
    echo [ERROR] requirements.txt missing
    set ready=0
)

if exist ".streamlit\config.toml" (
    echo [OK] Streamlit config ready
) else (
    echo [ERROR] Streamlit config missing
    set ready=0
)

if exist "app_final.py" (
    echo [OK] Main app file ready
) else (
    echo [ERROR] Main app file missing
    set ready=0
)

if exist ".gitignore" (
    echo [OK] .gitignore configured
) else (
    echo [WARNING] .gitignore needs attention
)

echo.
if %ready%==1 (
    echo ========================================
    echo   READY FOR DEPLOYMENT!
    echo ========================================
    echo.
    echo Next steps:
    echo.
    echo 1. Push to GitHub:
    echo    git add .
    echo    git commit -m "Ready for deployment"
    echo    git push origin main
    echo.
    echo 2. Deploy on Streamlit Cloud:
    echo    - Go to: https://share.streamlit.io
    echo    - Click 'New app'
    echo    - Select your repo and app_final.py
    echo    - Add secrets ^(GITHUB_TOKEN, HUGGINGFACEHUB_API_TOKEN^)
    echo    - Click 'Deploy!'
    echo.
    echo 3. Follow DEPLOYMENT_CHECKLIST.md for detailed steps
    echo.
) else (
    echo ========================================
    echo   NOT READY - Please fix issues above
    echo ========================================
)

echo.
echo Press any key to exit...
pause >nul
