@echo off
cd /d D:\E\skin_d-main\skin_d-main
call .venv\Scripts\activate
set PYTHONPATH=src
set CHECKPOINT_PATH=models\best.pt
set CONFIG_PATH=configs\default.yaml
python -m uvicorn skin_disease_ai.api.main:app
pause