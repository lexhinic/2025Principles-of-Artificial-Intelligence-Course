# ui/main_window.py
import sys
import random
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSpinBox, QGroupBox, 
                             QFormLayout, QColorDialog, QScrollArea, 
                             QGridLayout, QFrame, QMessageBox, QSplitter)
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt5.QtCore import Qt, QSize

from game_logic import Game
from ui.board_widget import BoardWidget
try:
    from ui.style import get_style_sheet
except ImportError:
    def get_style_sheet():
        return ""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.game = Game()
        
        self.initUI()
    
    def initUI(self):
        """初始化UI"""
        self.setWindowTitle('交叉线游戏')
        self.setMinimumSize(900, 600)
        
        # 应用样式表
        self.setStyleSheet(get_style_sheet())
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建分隔器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        
        # 设置面板
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(10)
        
        # 棋盘大小设置
        board_size_group = QGroupBox("棋盘设置")
        board_size_layout = QFormLayout()
        board_size_layout.setSpacing(10)
        
        self.board_size_spin = QSpinBox()
        self.board_size_spin.setRange(3, 15)
        self.board_size_spin.setValue(self.game.board.size)
        self.board_size_spin.setFixedHeight(25)
        
        self.create_board_btn = QPushButton("创建棋盘")
        self.create_board_btn.clicked.connect(self.create_board)
        self.create_board_btn.setFixedHeight(30)
        
        board_size_layout.addRow("棋盘大小:", self.board_size_spin)
        board_size_layout.addRow(self.create_board_btn)
        board_size_group.setLayout(board_size_layout)
        
        # 棋子设置
        pieces_group = QGroupBox("棋子设置")
        pieces_layout = QVBoxLayout()
        pieces_layout.setSpacing(10)
        
        # 添加棋子表单
        add_piece_form = QGridLayout()
        add_piece_form.setSpacing(8)
        
        self.piece_color_btn = QPushButton("选择颜色")
        self.piece_color_btn.setStyleSheet(f"background-color: #3498db; color: white;")
        self.current_color = "#3498db"
        self.piece_color_btn.clicked.connect(self.choose_piece_color)
        self.piece_color_btn.setFixedHeight(30)
        
        add_piece_form.addWidget(QLabel("颜色:"), 0, 0)
        add_piece_form.addWidget(self.piece_color_btn, 0, 1, 1, 3)
        
        add_piece_form.addWidget(QLabel("第一个棋子位置:"), 1, 0, 1, 4)
        
        row_label1 = QLabel("行:")
        row_label1.setFixedWidth(30)
        add_piece_form.addWidget(row_label1, 2, 0)
        
        self.piece1_row_spin = QSpinBox()
        self.piece1_row_spin.setRange(0, self.game.board.size - 1)
        self.piece1_row_spin.setFixedHeight(25)
        add_piece_form.addWidget(self.piece1_row_spin, 2, 1)
        
        col_label1 = QLabel("列:")
        col_label1.setFixedWidth(30)
        add_piece_form.addWidget(col_label1, 2, 2)
        
        self.piece1_col_spin = QSpinBox()
        self.piece1_col_spin.setRange(0, self.game.board.size - 1)
        self.piece1_col_spin.setFixedHeight(25)
        add_piece_form.addWidget(self.piece1_col_spin, 2, 3)
        
        add_piece_form.addWidget(QLabel("第二个棋子位置:"), 3, 0, 1, 4)
        
        row_label2 = QLabel("行:")
        row_label2.setFixedWidth(30)
        add_piece_form.addWidget(row_label2, 4, 0)
        
        self.piece2_row_spin = QSpinBox()
        self.piece2_row_spin.setRange(0, self.game.board.size - 1)
        self.piece2_row_spin.setFixedHeight(25)
        add_piece_form.addWidget(self.piece2_row_spin, 4, 1)
        
        col_label2 = QLabel("列:")
        col_label2.setFixedWidth(30)
        add_piece_form.addWidget(col_label2, 4, 2)
        
        self.piece2_col_spin = QSpinBox()
        self.piece2_col_spin.setRange(0, self.game.board.size - 1)
        self.piece2_col_spin.setFixedHeight(25)
        add_piece_form.addWidget(self.piece2_col_spin, 4, 3)
        
        self.add_piece_btn = QPushButton("添加棋子对")
        self.add_piece_btn.clicked.connect(self.add_piece)
        self.add_piece_btn.setFixedHeight(30)
        
        # 棋子列表
        pieces_list_label = QLabel("已添加的棋子对:")
        pieces_list_label.setFont(QFont("Arial", 9, QFont.Bold))
        
        self.pieces_list_widget = QWidget()
        self.pieces_list_layout = QVBoxLayout(self.pieces_list_widget)
        self.pieces_list_layout.setSpacing(5)
        self.pieces_list_layout.setContentsMargins(5, 5, 5, 5)
        
        pieces_scroll = QScrollArea()
        pieces_scroll.setWidgetResizable(True)
        pieces_scroll.setWidget(self.pieces_list_widget)
        pieces_scroll.setMinimumHeight(120)
        
        self.clear_pieces_btn = QPushButton("清除所有棋子")
        self.clear_pieces_btn.clicked.connect(self.clear_pieces)
        self.clear_pieces_btn.setFixedHeight(30)
        self.clear_pieces_btn.setStyleSheet("background-color: #e67e22;")
        
        pieces_layout.addLayout(add_piece_form)
        pieces_layout.addWidget(self.add_piece_btn)
        pieces_layout.addWidget(pieces_list_label)
        pieces_layout.addWidget(pieces_scroll)
        pieces_layout.addWidget(self.clear_pieces_btn)
        
        pieces_group.setLayout(pieces_layout)
        
        # 搜索算法
        algorithm_group = QGroupBox("搜索算法")
        algorithm_layout = QVBoxLayout()
        algorithm_layout.setSpacing(10)
        
        self.find_paths_btn = QPushButton("查找最短路径")
        self.find_paths_btn.clicked.connect(lambda: self.find_paths(False))
        self.find_paths_btn.setFixedHeight(30)
        
        self.find_paths_with_turns_btn = QPushButton("考虑转向代价的最短路径")
        self.find_paths_with_turns_btn.clicked.connect(lambda: self.find_paths(True))
        self.find_paths_with_turns_btn.setFixedHeight(30)
        
        self.clear_paths_btn = QPushButton("清除路径")
        self.clear_paths_btn.clicked.connect(self.clear_paths)
        self.clear_paths_btn.setFixedHeight(30)
        self.clear_paths_btn.setStyleSheet("background-color: #e67e22;")
        
        result_label = QLabel("结果:")
        result_label.setFont(QFont("Arial", 9, QFont.Bold))
        
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        result_frame.setObjectName("pieceFrame")
        result_frame_layout = QVBoxLayout(result_frame)
        
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setTextFormat(Qt.RichText)
        
        result_frame_layout.addWidget(self.result_label)
        
        algorithm_layout.addWidget(self.find_paths_btn)
        algorithm_layout.addWidget(self.find_paths_with_turns_btn)
        algorithm_layout.addWidget(self.clear_paths_btn)
        algorithm_layout.addWidget(result_label)
        algorithm_layout.addWidget(result_frame)
        
        algorithm_group.setLayout(algorithm_layout)
        
        # 添加到设置面板
        settings_layout.addWidget(board_size_group)
        settings_layout.addWidget(pieces_group)
        settings_layout.addWidget(algorithm_group)
        settings_layout.addStretch()
        
        # 棋盘部分
        board_panel = QGroupBox("游戏棋盘")
        board_layout = QVBoxLayout(board_panel)
        
        self.board_widget = BoardWidget(self.game)
        self.board_widget.cell_clicked.connect(self.handle_cell_click)
        
        board_layout.addWidget(self.board_widget)
        
        # 添加到分隔器
        splitter.addWidget(settings_panel)
        splitter.addWidget(board_panel)
        splitter.setSizes([300, 600])  # 设置初始大小比例
        
        # 添加到主布局
        main_layout.addWidget(splitter)
        
        # 更新UI
        self.update_pieces_list()
        
    def create_board(self):
        """创建新棋盘"""
        size = self.board_size_spin.value()
        success, message = self.game.resize_board(size)
        
        if success:
            # 更新棋子坐标输入范围
            self.piece1_row_spin.setRange(0, size - 1)
            self.piece1_col_spin.setRange(0, size - 1)
            self.piece2_row_spin.setRange(0, size - 1)
            self.piece2_col_spin.setRange(0, size - 1)
            
            # 更新棋盘显示
            self.board_widget.update_board()
            
            # 清除结果
            self.result_label.setText("")
        else:
            QMessageBox.warning(self, "创建棋盘", message)
    
    def choose_piece_color(self):
        """选择棋子颜色"""
        color = QColorDialog.getColor(QColor(self.current_color), self, "选择棋子颜色")
        
        if color.isValid():
            self.current_color = color.name()
            self.piece_color_btn.setStyleSheet(f"background-color: {self.current_color}; color: white;")
    
    def add_piece(self):
        """添加棋子对"""
        pos1 = (self.piece1_row_spin.value(), self.piece1_col_spin.value())
        pos2 = (self.piece2_row_spin.value(), self.piece2_col_spin.value())
        
        success, result = self.game.add_piece_pair(self.current_color, pos1, pos2)
        
        if success:
            # 更新棋盘
            self.board_widget.update_board()
            
            # 更新棋子列表
            self.update_pieces_list()
            
            # 清空输入
            self.piece1_row_spin.setValue(0)
            self.piece1_col_spin.setValue(0)
            self.piece2_row_spin.setValue(0)
            self.piece2_col_spin.setValue(0)
            
            # 随机生成新颜色
            self.current_color = self.get_random_color()
            self.piece_color_btn.setStyleSheet(f"background-color: {self.current_color}; color: white;")
        else:
            QMessageBox.warning(self, "添加棋子", result)
    
    def get_random_color(self):
        """生成随机颜色"""
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    def update_pieces_list(self):
        """更新棋子列表"""
        # 清除现有项目
        while self.pieces_list_layout.count():
            child = self.pieces_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # 添加新项目
        for piece in self.game.pieces:
            piece_frame = QFrame()
            piece_frame.setObjectName("pieceFrame")
            piece_frame.setFrameShape(QFrame.StyledPanel)
            piece_frame.setLineWidth(1)
            
            piece_layout = QHBoxLayout(piece_frame)
            piece_layout.setContentsMargins(5, 5, 5, 5)
            
            # 颜色标识
            color_label = QLabel()
            color_label.setFixedSize(20, 20)
            color_label.setStyleSheet(f"background-color: {piece['color']}; border-radius: 10px;")
            
            # ID标签
            id_label = QLabel(f"{chr(64 + piece['id'])}")
            id_label.setFixedWidth(20)
            id_label.setAlignment(Qt.AlignCenter)
            id_label.setFont(QFont("Arial", 9, QFont.Bold))
            
            # 坐标信息
            info_label = QLabel(f"({piece['positions'][0][0]},{piece['positions'][0][1]}) - "
                               f"({piece['positions'][1][0]},{piece['positions'][1][1]})")
            info_label.setFont(QFont("Arial", 9))
            
            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.setFixedWidth(50)
            delete_btn.setFixedHeight(25)
            delete_btn.setStyleSheet("background-color: #e74c3c;")
            delete_btn.clicked.connect(lambda checked, pid=piece['id']: self.remove_piece(pid))
            
            piece_layout.addWidget(color_label)
            piece_layout.addWidget(id_label)
            piece_layout.addWidget(info_label, 1)
            piece_layout.addWidget(delete_btn)
            
            self.pieces_list_layout.addWidget(piece_frame)
        
        # 添加拉伸项，确保列表项目靠上显示
        self.pieces_list_layout.addStretch()
    
    def remove_piece(self, piece_id):
        """删除棋子对"""
        self.game.remove_piece_pair(piece_id)
        self.update_pieces_list()
        self.board_widget.update_board()
        self.result_label.setText("")
    
    def clear_pieces(self):
        """清除所有棋子"""
        self.game.clear_pieces()
        self.update_pieces_list()
        self.board_widget.update_board()
        self.result_label.setText("")
    
    def handle_cell_click(self, row, col):
        """处理棋盘单元格点击事件"""
        # 如果第一个棋子位置未设置，则设置第一个棋子位置
        if self.piece1_row_spin.value() == 0 and self.piece1_col_spin.value() == 0:
            self.piece1_row_spin.setValue(row)
            self.piece1_col_spin.setValue(col)
        # 否则设置第二个棋子位置
        else:
            self.piece2_row_spin.setValue(row)
            self.piece2_col_spin.setValue(col)
    
    def find_paths(self, consider_turns):
        """查找路径"""
        # 检查是否有棋子
        if not self.game.pieces:
            QMessageBox.warning(self, "查找路径", "请先添加棋子")
            return
        
        # 查找路径
        results = self.game.find_all_paths(consider_turns)
        
        # 显示结果
        result_text = "<style>span.success{color:#2ecc71;font-weight:bold;} span.fail{color:#e74c3c;font-weight:bold;}</style>"
        
        for result in results["results"]:
            piece_id = result["piece_id"]
            piece_char = chr(64 + piece_id)
            color = result["color"]
            
            if result["success"]:
                if consider_turns:
                    result_text += (f"<p><span style='color:{color}'>■ {piece_char}号棋子对</span>的路径: "
                                  f"长度=<b>{result['path_length']}</b>, "
                                  f"转向次数=<b>{result['turns']}</b>, "
                                  f"总代价=<b>{result['total_cost']}</b></p>")
                else:
                    result_text += f"<p><span style='color:{color}'>■ {piece_char}号棋子对</span>的最短路径长度: <b>{result['path_length']}</b></p>"
            else:
                result_text += f"<p><span style='color:{color}'>■ {piece_char}号棋子对</span><span class='fail'>无法找到有效路径</span></p>"
        
        # 添加总体结果
        status = "<p><span class='success'>成功：所有棋子对都成功连接！</span></p>" if results["all_connected"] else "<p><span class='fail'>失败：无法完成所有棋子的连接</span></p>"
        result_text += f"{status}"
        
        self.result_label.setText(result_text)
        
        # 更新棋盘显示路径
        self.board_widget.update_board()
    
    def clear_paths(self):
        """清除路径"""
        self.game.clear_paths()
        self.board_widget.update_board()
        self.result_label.setText("")