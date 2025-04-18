# yolov8

# USE PYTHON version 3.9/3.10 in order to EXPORT the model as ".torchscript"
- IF YOU ENCOUNTER THIS ERROR: `Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"`: https://visualstudio.microsoft.com/visual-cpp-build-tools/ 
- WATCH THIS: https://www.youtube.com/watch?v=gzRBH726vUs
- tldr: `pip install cmake`, INSTALL: Visual Studio Installer > Modify > `Desktop development with C++` > Modify

### creating the virtual environment
```
python -m venv venv
```

### activating the virtual environment: (run in command prompt)
```
cd venv/Scripts
activate.bat
```

### install ultaralytics directly
```
pip install ultralytics
```

OR 

### installing packages from the requirements.txt
```
pip install -r requirements.txt
```

### start the training 
- (make sure you are in the correct folder location. and venv is activated)
```
python app.py
```

### gitignore must include

```
venv
runs
dataset
```

### creating requirements.txt
```
pip freeze > requirements.txt
```
