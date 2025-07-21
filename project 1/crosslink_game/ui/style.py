# ui/style.py

def get_style_sheet():
    """获取应用程序的样式表"""
    return """
    QMainWindow {
        background-color: #f5f5f7;
    }
    
    QGroupBox {
        background-color: #ffffff;
        border-radius: 6px;
        font-weight: bold;
        margin-top: 12px;
        border: 1px solid #e0e0e0;
        padding-top: 12px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #333333;
    }
    
    QPushButton {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
    }
    
    QPushButton:hover {
        background-color: #3a80d2;
    }
    
    QPushButton:pressed {
        background-color: #2a70c2;
    }
    
    QSpinBox {
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 4px;
        background: white;
    }
    
    QLabel {
        color: #333333;
    }
    
    QScrollArea {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background: white;
    }
    
    QFrame#pieceFrame {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 5px;
        margin-bottom: 5px;
        background-color: #f9f9f9;
    }
    """