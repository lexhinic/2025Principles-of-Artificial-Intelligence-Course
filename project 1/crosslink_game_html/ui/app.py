# ui/app.py
from flask import Flask, render_template, request, jsonify
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_logic import Game

app = Flask(__name__)
game = Game()

@app.route('/')
def index():
    """渲染游戏主页"""
    return render_template('index.html')

@app.route('/api/create_board', methods=['POST'])
def create_board():
    """创建或调整棋盘大小"""
    data = request.json
    size = int(data.get('size', 8))
    success, message = game.resize_board(size)
    
    return jsonify({
        'success': success,
        'message': message,
        'game_state': game.get_game_state()
    })

@app.route('/api/add_piece', methods=['POST'])
def add_piece():
    """添加一对棋子"""
    data = request.json
    color = data.get('color')
    pos1 = (int(data.get('pos1')[0]), int(data.get('pos1')[1]))
    pos2 = (int(data.get('pos2')[0]), int(data.get('pos2')[1]))
    
    success, result = game.add_piece_pair(color, pos1, pos2)
    
    return jsonify({
        'success': success,
        'message': result if not success else f"棋子对 {result} 添加成功",
        'piece_id': result if success else None,
        'game_state': game.get_game_state()
    })

@app.route('/api/remove_piece', methods=['POST'])
def remove_piece():
    """删除一对棋子"""
    data = request.json
    piece_id = int(data.get('piece_id'))
    
    success = game.remove_piece_pair(piece_id)
    
    return jsonify({
        'success': success,
        'message': f"棋子对 {piece_id} 删除成功" if success else f"未找到棋子对 {piece_id}",
        'game_state': game.get_game_state()
    })

@app.route('/api/clear_pieces', methods=['POST'])
def clear_pieces():
    """清除所有棋子"""
    game.clear_pieces()
    
    return jsonify({
        'success': True,
        'message': "所有棋子已清除",
        'game_state': game.get_game_state()
    })

@app.route('/api/find_paths', methods=['POST'])
def find_paths():
    """查找最短路径"""
    data = request.json
    consider_turns = data.get('consider_turns', False)
    
    results = game.find_all_paths(consider_turns)
    
    return jsonify({
        'success': True,
        'results': results,
        'game_state': game.get_game_state()
    })

@app.route('/api/clear_paths', methods=['POST'])
def clear_paths():
    """清除所有路径"""
    game.clear_paths()
    
    return jsonify({
        'success': True,
        'message': "所有路径已清除",
        'game_state': game.get_game_state()
    })

def run_server():
    """启动Web服务器"""
    app.run(debug=True)

if __name__ == '__main__':
    run_server()