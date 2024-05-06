mkdir venv
python -m venv ./venv
venv\Scripts\pip.exe install kivy matplotlib sympy kivy-garden
venv\Scripts\pip.exe install --upgrade setuptools
venv\Scripts\pip.exe install https://github.com/kivy-garden/matplotlib/archive/master.zip