# ui/board_widget.py
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRectF, QPointF

class BoardWidget(QWidget):
    """棋盘控件"""
    cell_clicked = pyqtSignal(int, int)  
    
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.hover_cell = None
    
    def sizeHint(self):
        """建议大小"""
        return QSize(500, 500)
    
    def paintEvent(self, event):
        """绘制棋盘"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        board_size = self.game.board.size
        
        # 计算单元格大小
        widget_size = min(self.width(), self.height())
        cell_size = widget_size / board_size
        
        # 绘制棋盘背景
        painter.fillRect(self.rect(), QColor("#f0f0f0"))
        
        # 绘制棋盘
        for row in range(board_size):
            for col in range(board_size):
                # 绘制单元格
                rect_x = col * cell_size
                rect_y = row * cell_size
                
                # 棋盘格子颜色交替
                if (row + col) % 2 == 0:
                    cell_color = QColor(245, 245, 245)
                else:
                    cell_color = QColor(235, 235, 235)
                
                # 高亮悬停单元格
                if self.hover_cell == (row, col):
                    cell_color = QColor(220, 235, 250)
                
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                painter.setBrush(QBrush(cell_color))
                painter.drawRect(QRectF(rect_x, rect_y, cell_size, cell_size))
                
                # 绘制坐标
                painter.setPen(QColor(150, 150, 150))
                painter.setFont(QFont("Arial", 8))
                painter.drawText(
                    QRectF(rect_x + 2, rect_y + 2, cell_size - 4, cell_size / 4),
                    Qt.AlignLeft | Qt.AlignTop,
                    f"{row},{col}"
                )
        
        # 绘制路径
        for i, path in enumerate(self.game.paths):
            if i < len(self.game.pieces):
                piece = self.game.pieces[i]
                color = QColor(piece["color"])
                shadow_color = QColor(0, 0, 0, 40) 
                
                # 路径宽度
                path_width = cell_size * 0.2
                
                # 绘制路径阴影
                painter.setPen(QPen(shadow_color, path_width + 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                for j in range(len(path) - 1):
                    start_row, start_col = path[j]
                    end_row, end_col = path[j+1]
                    
                    start_x = (start_col + 0.5) * cell_size
                    start_y = (start_row + 0.5) * cell_size
                    end_x = (end_col + 0.5) * cell_size
                    end_y = (end_row + 0.5) * cell_size
                    
                    painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
                
                # 绘制实际路径
                painter.setPen(QPen(color, path_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                
                for j in range(len(path) - 1):
                    start_row, start_col = path[j]
                    end_row, end_col = path[j+1]
                    
                    start_x = (start_col + 0.5) * cell_size
                    start_y = (start_row + 0.5) * cell_size
                    end_x = (end_col + 0.5) * cell_size
                    end_y = (end_row + 0.5) * cell_size
                    
                    painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
                
                # 在转弯处添加标记
                if len(path) > 2:
                    for j in range(1, len(path) - 1):
                        prev_row, prev_col = path[j-1]
                        curr_row, curr_col = path[j]
                        next_row, next_col = path[j+1]
                        
                        prev_dir = (curr_row - prev_row, curr_col - prev_col)
                        next_dir = (next_row - curr_row, next_col - curr_col)
                        
                        # 如果方向变化了，绘制转弯点标记
                        if prev_dir != next_dir:
                            center_x = (curr_col + 0.5) * cell_size
                            center_y = (curr_row + 0.5) * cell_size
                            turn_radius = cell_size * 0.1
                            
                            # 绘制转弯点标记
                            painter.setBrush(QBrush(Qt.white))
                            painter.setPen(QPen(color, 1))
                            painter.drawEllipse(QPointF(center_x, center_y), turn_radius, turn_radius)
        
        # 绘制棋子
        for piece in self.game.pieces:
            color = QColor(piece["color"])
            label = chr(64 + piece["id"])
            
            for pos in piece["positions"]:
                row, col = pos
                
                center_x = (col + 0.5) * cell_size
                center_y = (row + 0.5) * cell_size
                radius = cell_size * 0.4
                
                # 绘制棋子阴影
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 50)))
                painter.drawEllipse(QRectF(center_x - radius + 2, center_y - radius + 2, radius * 2, radius * 2))
                
                # 绘制棋子渐变背景
                gradient = QLinearGradient(center_x - radius, center_y - radius, center_x + radius, center_y + radius)
                lighter_color = QColor(color).lighter(120)
                darker_color = QColor(color).darker(120)
                
                gradient.setColorAt(0, lighter_color)
                gradient.setColorAt(1, darker_color)
                
                painter.setPen(QPen(Qt.black, 0.5))
                painter.setBrush(QBrush(gradient))
                painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))
                
                # 绘制标签
                painter.setPen(Qt.white)
                painter.setFont(QFont("Arial", int(radius * 0.8), QFont.Bold))
                painter.drawText(
                    QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2), 
                    Qt.AlignCenter, 
                    label
                )
    
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        # 计算鼠标悬停的单元格
        board_size = self.game.board.size
        widget_size = min(self.width(), self.height())
        cell_size = widget_size / board_size
        
        col = int(event.x() / cell_size)
        row = int(event.y() / cell_size)
        
        # 检查是否在棋盘范围内
        if 0 <= row < board_size and 0 <= col < board_size:
            if self.hover_cell != (row, col):
                self.hover_cell = (row, col)
                self.update()  # 重绘界面
        else:
            if self.hover_cell is not None:
                self.hover_cell = None
                self.update()  # 重绘界面
    
    def leaveEvent(self, event):
        """鼠标离开事件"""
        if self.hover_cell is not None:
            self.hover_cell = None
            self.update()  # 重绘界面
    
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            # 计算点击的单元格
            board_size = self.game.board.size
            widget_size = min(self.width(), self.height())
            cell_size = widget_size / board_size
            
            col = int(event.x() / cell_size)
            row = int(event.y() / cell_size)
            
            # 检查是否在棋盘范围内
            if 0 <= row < board_size and 0 <= col < board_size:
                self.cell_clicked.emit(row, col)
    
    def update_board(self):
        """更新棋盘"""
        self.update()