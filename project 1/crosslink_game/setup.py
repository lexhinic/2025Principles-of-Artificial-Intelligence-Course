import sys
import os
from PyInstaller.__main__ import run

if __name__ == '__main__':
    sys.argv = [
        'PyInstaller',
        '--name=交叉线游戏',
        '--windowed',  
        '--onedir',    
        '--add-data=ui;ui',  
        '--hidden-import=PyQt5',  
        '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtGui',
        '--log-level=DEBUG',  
        'main.py'
    ]
    run()