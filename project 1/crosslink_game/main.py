# main.py
import sys
import traceback
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow

def exception_hook(exctype, value, tb):
    """异常处理钩子，捕获并显示未处理的异常"""
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    
    # 将错误写入日志文件
    log_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    log_path = os.path.join(log_dir, 'error_log.txt')
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{'='*50}\n")
        f.write(f"Error occurred at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(error_msg)
        f.write(f"{'='*50}\n\n")
    
    # 显示错误消息框
    if QApplication.instance():
        QMessageBox.critical(None, "错误", 
                             f"程序发生错误:\n{str(value)}\n\n详细信息已写入: {log_path}")
    
    # 调用原始的异常处理器
    sys.__excepthook__(exctype, value, tb)

def main():
    # 设置高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # 设置钩子
    sys.excepthook = exception_hook
    
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  
        
        # 设置字体
        font = QFont("Arial", 9)
        app.setFont(font)
        
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        # 捕获所有异常，确保被钩子处理
        exception_hook(type(e), e, e.__traceback__)

if __name__ == "__main__":
    main()