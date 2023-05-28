# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 32:
# 99309 Rafael Girão
# 104147 Guilherme Marcondes

import sys
from enum import Enum
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)
import numpy as np

DEBUG = True


class BoardPosition(Enum):
    TOP = "t"
    BOTTOM = "b"
    LEFT = "l"
    RIGHT = "r"
    MIDDLE = "m"
    CENTER = "c"
    WATER = "."
    UNKNOWN = "?" if DEBUG else " "


class HintPosition(Enum):
    TOP = "T"
    BOTTOM = "B"
    LEFT = "L"
    RIGHT = "R"
    MIDDLE = "M"
    CENTER = "C"
    WATER = "W"


class PlaceDirection(Enum):
    LEFT_TO_RIGHT = 1
    TOP_TO_BOTTOM = 2
    RIGHT_TO_LEFT = 3
    BOTTOM_TO_TOP = 4

    def is_horizontal(self) -> bool:
        return self == PlaceDirection.LEFT_TO_RIGHT or self == PlaceDirection.RIGHT_TO_LEFT

    def is_vertical(self) -> bool:
        return self == PlaceDirection.TOP_TO_BOTTOM or self == PlaceDirection.BOTTOM_TO_TOP


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(
        self,
        positions: np.array = np.full((10, 10), BoardPosition.UNKNOWN),
        row_pieces: np.array = None,
        col_pieces: np.array = None,
    ):
        """Inicializa um tabuleiro vazio."""
        self.positions = positions
        self.row_pieces = row_pieces
        self.col_pieces = col_pieces

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if row < 0 or row > 9 or col < 0 or col > 9:
            return BoardPosition.UNKNOWN
        return self.positions[row][col]

    def set_value(self, row: int, col: int, value: BoardPosition) -> None:
        """Define o valor na respetiva posição do tabuleiro."""
        if row < 0 or row > 9 or col < 0 or col > 9:
            return
        self.positions[row][col] = value

    def set_line(
        self,
        row: int,
        col: int,
        size: int,
        value: BoardPosition,
        direction: PlaceDirection,
    ):
        """Define o valor na respetiva linha do tabuleiro."""
        match direction:
            case PlaceDirection.LEFT_TO_RIGHT:
                for i in range(size):
                    self.set_value(row, col + i, value)
            case PlaceDirection.TOP_TO_BOTTOM:
                for i in range(size):
                    self.set_value(row + i, col, value)
            case PlaceDirection.RIGHT_TO_LEFT:
                for i in range(size):
                    self.set_value(row, col - i, value)
            case PlaceDirection.BOTTOM_TO_TOP:
                for i in range(size):
                    self.set_value(row - i, col, value)

    def place_boat(self, row: int, col: int, size: int, direction: PlaceDirection):
        """Posiciona um barco ao tabuleiro."""

        # Coloca o barco no tabuleiro
        self.set_line(row, col, size, BoardPosition.CENTER, direction)

        if size == 1:
            return

        # Coloca as extremidades do barco
        match direction:
            case PlaceDirection.LEFT_TO_RIGHT:
                self.set_value(row, col, BoardPosition.LEFT)
                self.set_value(row, col + size - 1, BoardPosition.RIGHT)
            case PlaceDirection.TOP_TO_BOTTOM:
                self.set_value(row, col, BoardPosition.TOP)
                self.set_value(row + size - 1, col, BoardPosition.BOTTOM)
            case PlaceDirection.RIGHT_TO_LEFT:
                self.set_value(row, col, BoardPosition.RIGHT)
                self.set_value(row, col - size + 1, BoardPosition.LEFT)
            case PlaceDirection.BOTTOM_TO_TOP:
                self.set_value(row, col, BoardPosition.BOTTOM)
                self.set_value(row - size + 1, col, BoardPosition.TOP)

    def place_boat_and_waters(self, row: int, col: int, size: int, direction: PlaceDirection):
        """Coloca agua ao redor de um barco."""

        # Coloca agua ao redor do barco
        match direction:
            case PlaceDirection.LEFT_TO_RIGHT:
                self.set_line(row - 1, col - 1, size + 2, BoardPosition.WATER, direction)
                self.set_line(row, col - 1, size + 2, BoardPosition.WATER, direction)
                self.set_line(row + 1, col - 1, size + 2, BoardPosition.WATER, direction)
            case PlaceDirection.TOP_TO_BOTTOM:
                self.set_line(row - 1, col - 1, size + 2, BoardPosition.WATER, direction)
                self.set_line(row - 1, col, size + 2, BoardPosition.WATER, direction)
                self.set_line(row - 1, col + 1, size + 2, BoardPosition.WATER, direction)
            case PlaceDirection.RIGHT_TO_LEFT:
                self.set_line(row - 1, col + 1, size + 2, BoardPosition.WATER, direction)
                self.set_line(row, col + 1, size + 2, BoardPosition.WATER, direction)
                self.set_line(row + 1, col + 1, size + 2, BoardPosition.WATER, direction)
            case PlaceDirection.BOTTOM_TO_TOP:
                self.set_line(row + 1, col - 1, size + 2, BoardPosition.WATER, direction)
                self.set_line(row + 1, col, size + 2, BoardPosition.WATER, direction)
                self.set_line(row + 1, col + 1, size + 2, BoardPosition.WATER, direction)

        # Coloca o barco
        self.place_boat(row, col, size, direction)

    def adjacent_vertical_values(self, row: int, col: int) -> tuple[str, str]:
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        return (self.get_value(row - 1, col), self.get_value(row + 1, col))

    def adjacent_horizontal_values(self, row: int, col: int) -> tuple[str, str]:
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_value(row, col - 1), self.get_value(row, col + 1))

    def adjacent_diagonal_values(self, row: int, col: int) -> tuple[str, str, str, str]:
        """Devolve os valores das diagonais."""
        return (
            self.get_value(row - 1, col - 1),
            self.get_value(row + 1, col + 1),
            self.get_value(row + 1, col - 1),
            self.get_value(row - 1, col + 1),
        )

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board."""

        positions = np.full((10, 10), BoardPosition.UNKNOWN)
        row_pieces, col_pieces = Board.parse_pieces()

        board = Board(positions, row_pieces, col_pieces)

        hint_count = int(sys.stdin.readline())
        for _ in range(hint_count):
            line = sys.stdin.readline().split()
            row = int(line[1])
            col = int(line[2])
            value = HintPosition(line[3])

            match value:
                case HintPosition.WATER:
                    board.set_value(row, col, HintPosition.WATER)
                case HintPosition.CENTER:
                    board.place_boat_and_waters(row, col, 1, PlaceDirection.LEFT_TO_RIGHT)
                    board.set_value(row, col, HintPosition.CENTER)
                case HintPosition.TOP:
                    board.set_line(
                        row - 1,
                        col - 1,
                        4,
                        BoardPosition.WATER,
                        PlaceDirection.TOP_TO_BOTTOM,
                    )
                    board.set_line(
                        row - 1,
                        col + 1,
                        4,
                        BoardPosition.WATER,
                        PlaceDirection.TOP_TO_BOTTOM,
                    )
                    board.set_value(row - 1, col, HintPosition.WATER)
                    board.set_value(row, col, HintPosition.TOP)
                case HintPosition.BOTTOM:
                    board.set_line(
                        row + 1,
                        col - 1,
                        4,
                        BoardPosition.WATER,
                        PlaceDirection.BOTTOM_TO_TOP,
                    )
                    board.set_line(
                        row + 1,
                        col + 1,
                        4,
                        BoardPosition.WATER,
                        PlaceDirection.BOTTOM_TO_TOP,
                    )
                    board.set_value(row + 1, col, HintPosition.WATER)
                    board.set_value(row, col, HintPosition.BOTTOM)
                case HintPosition.LEFT:
                    board.set_line(
                        row - 1,
                        col - 1,
                        4,
                        BoardPosition.WATER,
                        PlaceDirection.LEFT_TO_RIGHT,
                    )
                    board.set_line(
                        row + 1,
                        col - 1,
                        4,
                        BoardPosition.WATER,
                        PlaceDirection.LEFT_TO_RIGHT,
                    )
                    board.set_value(row, col - 1, HintPosition.WATER)
                    board.set_value(row, col, HintPosition.LEFT)
                case HintPosition.RIGHT:
                    board.set_line(
                        row - 1,
                        col + 1,
                        4,
                        BoardPosition.WATER,
                        PlaceDirection.RIGHT_TO_LEFT,
                    )
                    board.set_line(
                        row + 1,
                        col + 1,
                        4,
                        BoardPosition.WATER,
                        PlaceDirection.RIGHT_TO_LEFT,
                    )
                    board.set_value(row, col + 1, HintPosition.WATER)
                    board.set_value(row, col, HintPosition.RIGHT)

        return board

    @staticmethod
    def parse_pieces() -> tuple[np.array, np.array]:
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board."""
        line = sys.stdin.readline().split()
        row_pieces = np.array([int(x) for x in line if x != "ROW"])

        line = sys.stdin.readline().split()
        col_pieces = np.array([int(x) for x in line if x != "COLUMN"])

        return (row_pieces, col_pieces)

    def print(self):
        """Imprime o tabuleiro."""
        string = " " if DEBUG else ""
        for row in self.positions:
            print(string.join([str(i.value) for i in row]))


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    board = Board.parse_instance()
    board.print()
