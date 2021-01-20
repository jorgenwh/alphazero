import chess


board = chess.Board()

fen = board.fen()
print(board)
print()
print(fen)


board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
print()
print(board)