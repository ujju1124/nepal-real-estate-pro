@echo off
echo ============================================
echo Push to Hugging Face Spaces
echo ============================================
echo.

REM Prompt for HF username
set /p HF_USERNAME="Enter your Hugging Face username: "

REM Prompt for HF token
echo.
echo Get your token from: https://huggingface.co/settings/tokens
set /p HF_TOKEN="Enter your HF access token (write access): "

echo.
echo Adding HF remote...
git remote remove hf 2>nul
git remote add hf https://%HF_USERNAME%:%HF_TOKEN%@huggingface.co/spaces/%HF_USERNAME%/nepal-real-estate-pro

echo.
echo Pushing to HF Spaces (this may take 1-2 minutes for LFS files)...
git push hf master:main --force

echo.
echo ============================================
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS! Your app is deploying.
    echo.
    echo Check your Space at:
    echo https://huggingface.co/spaces/%HF_USERNAME%/nepal-real-estate-pro
    echo.
    echo Live app will be at:
    echo https://%HF_USERNAME%-nepal-real-estate-pro.hf.space
) else (
    echo FAILED! Check the error above.
    echo.
    echo Common issues:
    echo - Wrong username or token
    echo - Space name doesn't match
    echo - No write access on token
)
echo ============================================
echo.
pause
