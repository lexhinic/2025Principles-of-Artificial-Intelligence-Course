// ui/static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // DOM 元素
    const boardElement = document.getElementById('board');
    const boardSizeInput = document.getElementById('boardSize');
    const createBoardButton = document.getElementById('createBoard');
    const pieceSettings = document.getElementById('pieceSettings');
    const pieceColorInput = document.getElementById('pieceColor');
    const colorPreview = document.getElementById('colorPreview');
    const piece1RowInput = document.getElementById('piece1Row');
    const piece1ColInput = document.getElementById('piece1Col');
    const piece2RowInput = document.getElementById('piece2Row');
    const piece2ColInput = document.getElementById('piece2Col');
    const addPieceButton = document.getElementById('addPiece');
    const pieceListElement = document.getElementById('pieceList');
    const clearPiecesButton = document.getElementById('clearPieces');
    const findPathsButton = document.getElementById('findPaths');
    const findPathsWithTurnsButton = document.getElementById('findPathsWithTurns');
    const resultElement = document.getElementById('result');
    const pathResultsElement = document.getElementById('pathResults');
    const statusElement = document.getElementById('status');
    const clearPathsButton = document.getElementById('clearPaths');

    // 游戏状态
    let gameState = {
        boardSize: 8,
        pieces: [],
        paths: []
    };

    // 初始化棋盘
    function initializeBoard(size) {
        // 发送API请求创建棋盘
        fetch('/api/create_board', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ size: size }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState = data.game_state;
                
                // 更新输入限制
                const maxIndex = gameState.board_size - 1;
                piece1RowInput.max = maxIndex;
                piece1ColInput.max = maxIndex;
                piece2RowInput.max = maxIndex;
                piece2ColInput.max = maxIndex;
                
                // 清除结果
                resultElement.style.display = 'none';
                
                // 渲染棋盘
                renderBoard();
                
                // 显示棋子设置
                pieceSettings.style.display = 'block';
            } else {
                alert('创建棋盘失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('创建棋盘时出错:', error);
            alert('创建棋盘时出错');
        });
    }

    // 渲染棋盘
    function renderBoard() {
        boardElement.innerHTML = '';
        boardElement.style.gridTemplateColumns = `repeat(${gameState.board_size}, 40px)`;
        
        for (let row = 0; row < gameState.board_size; row++) {
            for (let col = 0; col < gameState.board_size; col++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.addEventListener('click', () => handleCellClick(row, col));
                boardElement.appendChild(cell);
            }
        }
        
        // 渲染棋子
        renderPieces();
        
        // 渲染路径
        renderPaths();
    }

    // 渲染棋子
    function renderPieces() {
        // 清除所有已存在的棋子
        document.querySelectorAll('.piece').forEach(p => p.remove());
        
        // 渲染新棋子
        gameState.pieces.forEach(piece => {
            piece.positions.forEach((pos, posIndex) => {
                const cell = getCellElement(pos[0], pos[1]);
                if (cell) {
                    const pieceElement = document.createElement('div');
                    pieceElement.className = 'piece';
                    pieceElement.style.backgroundColor = piece.color;
                    pieceElement.textContent = String.fromCharCode(65 + piece.id - 1); // A, B, C...
                    cell.appendChild(pieceElement);
                }
            });
        });
    }

    // 渲染路径
    function renderPaths() {
        // 清除所有已存在的路径
        document.querySelectorAll('.path').forEach(p => p.remove());
        
        // 找到对应的棋子颜色和路径
        if (!gameState.paths) return;
        
        gameState.paths.forEach((path, pathIndex) => {
            // 找到对应的棋子颜色
            const piece = gameState.pieces[pathIndex];
            if (!piece || !path) return;
            
            const color = piece.color;
            
            for (let i = 0; i < path.length - 1; i++) {
                const current = path[i];
                const next = path[i + 1];
                
                const cell = getCellElement(current[0], current[1]);
                if (!cell) continue;
                
                const pathElement = document.createElement('div');
                pathElement.className = 'path';
                pathElement.style.color = color;
                
                // 水平线
                if (current[0] === next[0]) {
                    pathElement.classList.add('horizontal-path');
                    if (current[1] < next[1]) {
                        // 向右
                        pathElement.style.width = '40px';
                        pathElement.style.left = '50%';
                    } else {
                        // 向左
                        pathElement.style.width = '40px';
                        pathElement.style.left = '0';
                    }
                }
                // 垂直线
                else if (current[1] === next[1]) {
                    pathElement.classList.add('vertical-path');
                    if (current[0] < next[0]) {
                        // 向下
                        pathElement.style.height = '40px';
                        pathElement.style.top = '50%';
                    } else {
                        // 向上
                        pathElement.style.height = '40px';
                        pathElement.style.top = '0';
                    }
                }
                
                cell.appendChild(pathElement);
            }
        });
    }

    // 获取指定位置的单元格元素
    function getCellElement(row, col) {
        return document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
    }

    // 处理单元格点击事件 - 用于交互式添加棋子
    function handleCellClick(row, col) {
        // 如果是第一个棋子位置
        if (!piece1RowInput.value || !piece1ColInput.value) {
            piece1RowInput.value = row;
            piece1ColInput.value = col;
        }
        // 如果是第二个棋子位置
        else if (!piece2RowInput.value || !piece2ColInput.value) {
            piece2RowInput.value = row;
            piece2ColInput.value = col;
        }
    }

    // 添加棋子对
    function addPiece() {
        const color = pieceColorInput.value;
        const pos1 = [
            parseInt(piece1RowInput.value),
            parseInt(piece1ColInput.value)
        ];
        const pos2 = [
            parseInt(piece2RowInput.value),
            parseInt(piece2ColInput.value)
        ];
        
        // 验证输入
        if (isNaN(pos1[0]) || isNaN(pos1[1]) || isNaN(pos2[0]) || isNaN(pos2[1])) {
            alert('请输入有效的棋子位置');
            return;
        }
        
        // 发送API请求添加棋子
        fetch('/api/add_piece', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ color, pos1, pos2 }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState = data.game_state;
                
                // 更新棋子列表显示
                updatePieceList();
                
                // 渲染棋盘
                renderBoard();
                
                // 清除输入
                piece1RowInput.value = '';
                piece1ColInput.value = '';
                piece2RowInput.value = '';
                piece2ColInput.value = '';
                
                // 随机生成新颜色
                pieceColorInput.value = getRandomColor();
                colorPreview.style.backgroundColor = pieceColorInput.value;
            } else {
                alert('添加棋子失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('添加棋子时出错:', error);
            alert('添加棋子时出错');
        });
    }

    // 更新棋子列表显示
    function updatePieceList() {
        pieceListElement.innerHTML = '';
        
        gameState.pieces.forEach(piece => {
            const pieceItem = document.createElement('div');
            pieceItem.className = 'piece-item';
            
            const colorSpan = document.createElement('span');
            colorSpan.className = 'color-preview';
            colorSpan.style.backgroundColor = piece.color;
            
            pieceItem.appendChild(colorSpan);
            pieceItem.appendChild(document.createTextNode(
                `${String.fromCharCode(65 + piece.id - 1)}: (${piece.positions[0][0]},${piece.positions[0][1]}) - ` +
                `(${piece.positions[1][0]},${piece.positions[1][1]})`
            ));
            
            const removeButton = document.createElement('button');
            removeButton.textContent = '删除';
            removeButton.style.marginLeft = '10px';
            removeButton.style.padding = '2px 5px';
            removeButton.style.fontSize = '12px';
            removeButton.addEventListener('click', () => removePiece(piece.id));
            
            pieceItem.appendChild(removeButton);
            pieceListElement.appendChild(pieceItem);
        });
    }

    // 删除棋子对
    function removePiece(pieceId) {
        fetch('/api/remove_piece', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ piece_id: pieceId }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState = data.game_state;
                updatePieceList();
                renderBoard();
                
                // 清除结果
                resultElement.style.display = 'none';
            } else {
                alert('删除棋子失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('删除棋子时出错:', error);
            alert('删除棋子时出错');
        });
    }

    // 清除所有棋子
    function clearPieces() {
        fetch('/api/clear_pieces', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState = data.game_state;
                updatePieceList();
                renderBoard();
                
                // 清除结果
                resultElement.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('清除棋子时出错:', error);
            alert('清除棋子时出错');
        });
    }

    // 查找所有棋子对的最短路径
    function findPaths(considerTurns) {
        // 检查是否有棋子
        if (gameState.pieces.length === 0) {
            alert('请先添加棋子');
            return;
        }
        
        fetch('/api/find_paths', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ consider_turns: considerTurns }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState = data.game_state;
                
                // 显示结果
                const results = data.results;
                let resultText = '';
                
                results.results.forEach(result => {
                    const pieceId = result.piece_id;
                    const pieceChar = String.fromCharCode(65 + pieceId - 1);
                    
                    if (result.success) {
                        if (considerTurns) {
                            resultText += `${pieceChar}号棋子对的路径: 长度=${result.path_length}, 转向次数=${result.turns}, 总代价=${result.total_cost}<br>`;
                        } else {
                            resultText += `${pieceChar}号棋子对的最短路径长度: ${result.path_length}<br>`;
                        }
                    } else {
                        resultText += `${pieceChar}号棋子对无法找到有效路径<br>`;
                    }
                });
                
                pathResultsElement.innerHTML = resultText;
                statusElement.textContent = results.all_connected ? '成功：所有棋子对都成功连接！' : '失败：无法完成所有棋子的连接';
                statusElement.className = results.all_connected ? 'status success' : 'status failure';
                resultElement.style.display = 'block';
                
                // 渲染路径
                renderPaths();
            }
        })
        .catch(error => {
            console.error('查找路径时出错:', error);
            alert('查找路径时出错');
        });
    }

    // 清除路径
    function clearPaths() {
        fetch('/api/clear_paths', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState = data.game_state;
                renderBoard();
                resultElement.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('清除路径时出错:', error);
            alert('清除路径时出错');
        });
    }

    // 生成随机颜色
    function getRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    // 事件监听器
    createBoardButton.addEventListener('click', () => {
        const size = parseInt(boardSizeInput.value);
        if (isNaN(size) || size < 3 || size > 15) {
            alert('请输入有效的棋盘大小（3-15）');
            return;
        }
        initializeBoard(size);
    });

    pieceColorInput.addEventListener('input', () => {
        colorPreview.style.backgroundColor = pieceColorInput.value;
    });

    addPieceButton.addEventListener('click', addPiece);
    clearPiecesButton.addEventListener('click', clearPieces);
    findPathsButton.addEventListener('click', () => findPaths(false));
    findPathsWithTurnsButton.addEventListener('click', () => findPaths(true));
    clearPathsButton.addEventListener('click', clearPaths);

    // 初始化
    initializeBoard(8);
});