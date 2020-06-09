import chess

board = chess.Board()
# board.push_uci("d2d4")
x = board.is_legal(chess.Move.from_uci("D2D4".lower()))
y = board.is_legal(chess.Move.from_uci("D4D2".lower()))
print(x,y)