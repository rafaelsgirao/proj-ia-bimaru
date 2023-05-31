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
import copy as copy

DEBUG = False


def debug(msg):
    if DEBUG:
        print(f"DEBUG: {msg}")


class BoardPosition(Enum):
    TOP = "t"
    BOTTOM = "b"
    LEFT = "l"
    RIGHT = "r"
    MIDDLE = "m"
    CENTER = "c"
    WATER = "."
    UNKNOWN = "?" if DEBUG else " "
    OUT_OF_BOUNDS = "x"
    PROBABLY_BOAT = "p"


class HintPosition(Enum):
    TOP = "T"
    BOTTOM = "B"
    LEFT = "L"
    RIGHT = "R"
    MIDDLE = "M"
    CENTER = "C"
    WATER = "W"


def is_board_position(value: str) -> bool:
    """Verifica se o valor é uma posição do tabuleiro."""
    return value in (
        BoardPosition.BOTTOM,
        BoardPosition.TOP,
        BoardPosition.LEFT,
        BoardPosition.RIGHT,
        BoardPosition.MIDDLE,
        BoardPosition.CENTER,
        BoardPosition.PROBABLY_BOAT,
        HintPosition.TOP,
        HintPosition.BOTTOM,
        HintPosition.LEFT,
        HintPosition.RIGHT,
        HintPosition.MIDDLE,
        HintPosition.CENTER,
    )


def is_empty_position(pos: any) -> bool:
    return pos in (BoardPosition.UNKNOWN, BoardPosition.WATER, HintPosition.WATER, BoardPosition.OUT_OF_BOUNDS)


def is_placeble_position(pos: any) -> bool:
    return pos in (BoardPosition.UNKNOWN, BoardPosition.PROBABLY_BOAT)


class PlaceDirection(Enum):
    LEFT_TO_RIGHT = 1
    TOP_TO_BOTTOM = 2
    RIGHT_TO_LEFT = 3
    BOTTOM_TO_TOP = 4

    def is_horizontal(self) -> bool:
        return self is PlaceDirection.LEFT_TO_RIGHT or self is PlaceDirection.RIGHT_TO_LEFT

    def is_vertical(self) -> bool:
        return self is PlaceDirection.TOP_TO_BOTTOM or self is PlaceDirection.BOTTOM_TO_TOP


class RemainingBoats:
    """Representação interna dos barcos que faltam colocar."""

    def __init__(self, ones=4, twos=3, threes=2, fours=1):
        self.ones = ones
        self.twos = twos
        self.threes = threes
        self.fours = fours

    def get_next_size(self) -> int:
        """Devolve o tamanho do próximo barco a ser colocado."""

        if self.fours > 0:
            return 4
        if self.threes > 0:
            return 3
        if self.twos > 0:
            return 2
        if self.ones > 0:
            return 1
        return 0

    def get_values(self) -> (int, int, int, int):
        """Devolve os valores de barcos que faltam colocar."""
        return self.ones, self.twos, self.threes, self.fours

    def decrease_boat_count(self, size, count=1):
        """Diminui o número de barcos de um determinado tamanho."""
        if size == 1:
            self.ones -= count
        elif size == 2:
            self.twos -= count
        elif size == 3:
            self.threes -= count
        elif size == 4:
            self.fours -= count

    def copy(self):
        return RemainingBoats(self.ones, self.twos, self.threes, self.fours)


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def copy(self):
        return BimaruState(self.board.copy())


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(
        self,
        positions: np.array = np.full((10, 10), BoardPosition.UNKNOWN),
        row_pieces: np.array = np.array([]),
        col_pieces: np.array = np.array([]),
        remaining_boats: RemainingBoats = RemainingBoats(),
        already_placed_boats=[],
    ):
        """Inicializa um tabuleiro vazio."""
        self.positions = positions
        self.row_pieces = row_pieces
        self.col_pieces = col_pieces
        self.remaining_boats = remaining_boats
        self.already_placed_boats = already_placed_boats

    def incomplete_boats(self) -> int:
        """Devolve a quantidade de barcos incompletos."""
        count = 0
        for row in range(10):
            for col in range(10):
                value = self.get_value(row, col)
                if value in (HintPosition.TOP, HintPosition.LEFT, HintPosition.BOTTOM, HintPosition.RIGHT):
                    if value is HintPosition.TOP:
                        if self.get_value(row + 1, col) in (BoardPosition.UNKNOWN, BoardPosition.PROBABLY_BOAT):
                            count += 1
                    elif value is HintPosition.LEFT:
                        if self.get_value(row, col + 1) in (BoardPosition.UNKNOWN, BoardPosition.PROBABLY_BOAT):
                            count += 1
                    elif value is HintPosition.BOTTOM:
                        if self.get_value(row - 1, col) in (BoardPosition.UNKNOWN, BoardPosition.PROBABLY_BOAT):
                            count += 1
                    elif value is HintPosition.RIGHT:
                        if self.get_value(row, col - 1) in (BoardPosition.UNKNOWN, BoardPosition.PROBABLY_BOAT):
                            count += 1
        return count

    def copy(self):
        return Board(
            np.copy(self.positions),
            np.copy(self.row_pieces),
            np.copy(self.col_pieces),
            self.remaining_boats.copy(),
            self.already_placed_boats.copy()
        )

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if row < 0 or row > 9 or col < 0 or col > 9:
            return BoardPosition.OUT_OF_BOUNDS
        return self.positions[row][col]

    def set_value(self, row: int, col: int, value: BoardPosition) -> None:
        """Define o valor na respetiva posição do tabuleiro."""
        prev = self.get_value(row, col)
        if prev in HintPosition:
            if value.value == prev.value.lower() or (value is BoardPosition.WATER and prev is HintPosition.WATER):
                return
            raise ValueError(f"Cannot set value: {value} on a hint position: {prev}")
        if prev is BoardPosition.OUT_OF_BOUNDS:
            if value != BoardPosition.UNKNOWN and value != BoardPosition.WATER:
                raise ValueError(f"Cannot set value {value} in ({row},{col}) because the position is out of bounds")
            return
        if not is_board_position(prev) and is_board_position(value):
            if self.row_pieces[row] - 1 < 0 or self.col_pieces[col] - 1 < 0:
                raise ValueError(f"Cannot set value {value} in ({row},{col}) because the position is already full")
            self.row_pieces[row] -= 1
            self.col_pieces[col] -= 1
        self.positions[row][col] = value

    def set_line(self, row: int, col: int, size: int, value: BoardPosition, direction: PlaceDirection):
        """Define o valor na respetiva linha do tabuleiro."""
        if direction is PlaceDirection.LEFT_TO_RIGHT:
            for i in range(size):
                self.set_value(row, col + i, value)
        if direction is PlaceDirection.TOP_TO_BOTTOM:
            for i in range(size):
                self.set_value(row + i, col, value)
        if direction is PlaceDirection.RIGHT_TO_LEFT:
            for i in range(size):
                self.set_value(row, col - i, value)
        if direction is PlaceDirection.BOTTOM_TO_TOP:
            for i in range(size):
                self.set_value(row - i, col, value)

    def place_boat(self, row: int, col: int, size: int, direction: PlaceDirection):
        """Posiciona um barco ao tabuleiro."""

        if size == 1:
            self.set_value(row, col, BoardPosition.CENTER)
            return

        # Coloca o barco no tabuleiro
        if direction is PlaceDirection.LEFT_TO_RIGHT:
            self.set_value(row, col, BoardPosition.LEFT)
            self.set_value(row, col + size - 1, BoardPosition.RIGHT)
            self.set_line(row, col + 1, size - 2, BoardPosition.MIDDLE, direction)
        elif direction is PlaceDirection.TOP_TO_BOTTOM:
            self.set_value(row, col, BoardPosition.TOP)
            self.set_value(row + size - 1, col, BoardPosition.BOTTOM)
            self.set_line(row + 1, col, size - 2, BoardPosition.MIDDLE, direction)
        elif direction is PlaceDirection.RIGHT_TO_LEFT:
            self.set_value(row, col, BoardPosition.RIGHT)
            self.set_value(row, col - size + 1, BoardPosition.LEFT)
            self.set_line(row, col - 1, size - 2, BoardPosition.MIDDLE, direction)
        elif direction is PlaceDirection.BOTTOM_TO_TOP:
            self.set_value(row, col, BoardPosition.BOTTOM)
            self.set_value(row - size + 1, col, BoardPosition.TOP)
            self.set_line(row - 1, col, size - 2, BoardPosition.MIDDLE, direction)

        self.already_placed_boats.append((row, col, size, direction))

    def place_boat_and_waters(self, row: int, col: int, size: int, direction: PlaceDirection):
        """Coloca agua ao redor de um barco."""

        # Coloca agua ao redor do barco
        if direction is PlaceDirection.LEFT_TO_RIGHT:
            self.set_line(row - 1, col - 1, size + 2, BoardPosition.WATER, direction)
            self.set_value(row, col - 1, BoardPosition.WATER)
            self.set_value(row, col + size, BoardPosition.WATER)
            self.set_line(row + 1, col - 1, size + 2, BoardPosition.WATER, direction)
        if direction is PlaceDirection.TOP_TO_BOTTOM:
            self.set_line(row - 1, col - 1, size + 2, BoardPosition.WATER, direction)
            self.set_value(row - 1, col, BoardPosition.WATER)
            self.set_value(row + size, col, BoardPosition.WATER)
            self.set_line(row - 1, col + 1, size + 2, BoardPosition.WATER, direction)
        if direction is PlaceDirection.RIGHT_TO_LEFT:
            self.set_line(row - 1, col + 1, size + 2, BoardPosition.WATER, direction)
            self.set_value(row, col + 1, BoardPosition.WATER)
            self.set_value(row, col - size, BoardPosition.WATER)
            self.set_line(row + 1, col + 1, size + 2, BoardPosition.WATER, direction)
        if direction is PlaceDirection.BOTTOM_TO_TOP:
            self.set_line(row + 1, col - 1, size + 2, BoardPosition.WATER, direction)
            self.set_value(row + 1, col, BoardPosition.WATER)
            self.set_value(row - size, col, BoardPosition.WATER)
            self.set_line(row + 1, col + 1, size + 2, BoardPosition.WATER, direction)

        # Coloca o barco
        self.place_boat(row, col, size, direction)

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        return (self.get_value(row - 1, col), self.get_value(row + 1, col))

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_value(row, col - 1), self.get_value(row, col + 1))

    def adjacent_diagonal_values(self, row: int, col: int) -> (str, str, str, str):
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

            if value is HintPosition.WATER:
                board.set_value(row, col, HintPosition.WATER)
            if value is HintPosition.CENTER:
                board.place_boat_and_waters(row, col, 1, PlaceDirection.LEFT_TO_RIGHT)
                board.set_value(row, col, HintPosition.CENTER)
            if value is HintPosition.TOP:
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
                board.set_value(row - 1, col, BoardPosition.WATER)
                board.set_value(row, col, HintPosition.TOP)
                if board.get_value(row + 1, col) not in HintPosition:
                    board.set_value(row + 1, col, BoardPosition.PROBABLY_BOAT)
            if value is HintPosition.BOTTOM:
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
                board.set_value(row + 1, col, BoardPosition.WATER)
                board.set_value(row, col, HintPosition.BOTTOM)
                if board.get_value(row - 1, col) not in HintPosition:
                    board.set_value(row - 1, col, BoardPosition.PROBABLY_BOAT)
            if value is HintPosition.LEFT:
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
                board.set_value(row, col - 1, BoardPosition.WATER)
                board.set_value(row, col, HintPosition.LEFT)
                if board.get_value(row, col + 1) not in HintPosition:
                    board.set_value(row, col + 1, BoardPosition.PROBABLY_BOAT)
            if value is HintPosition.RIGHT:
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
                board.set_value(row, col + 1, BoardPosition.WATER)
                board.set_value(row, col, HintPosition.RIGHT)
                if board.get_value(row, col - 1) not in HintPosition:
                    board.set_value(row, col - 1, BoardPosition.PROBABLY_BOAT)
            if value is HintPosition.MIDDLE:
                board.set_value(row - 1, col - 1, BoardPosition.WATER)
                board.set_value(row - 1, col + 1, BoardPosition.WATER)
                board.set_value(row + 1, col - 1, BoardPosition.WATER)
                board.set_value(row + 1, col + 1, BoardPosition.WATER)
                board.set_value(row, col, HintPosition.MIDDLE)

        ones, twos, threes, fours = board.detect_boats()

        board.remaining_boats.decrease_boat_count(1, ones)
        board.remaining_boats.decrease_boat_count(2, twos)
        board.remaining_boats.decrease_boat_count(3, threes)
        board.remaining_boats.decrease_boat_count(4, fours)

        board.fill_empty_spaces()

        return board

    def have_enought_left_pieces_for_boat(self, row: int, col: int, size: int, direction: PlaceDirection) -> bool:
        if direction is PlaceDirection.LEFT_TO_RIGHT:
            if col + size > 10:
                return False
            left_row_pieces = self.row_pieces[row]
            for i in range(size):
                val = self.get_value(row, col + i)
                if self.col_pieces[col + i] == 0:
                    if val not in HintPosition and val is not BoardPosition.PROBABLY_BOAT:
                        return False
                left_row_pieces += 1 if val in HintPosition or val is BoardPosition.PROBABLY_BOAT else 0
            if left_row_pieces < size:
                return False
            return True
        elif direction is PlaceDirection.TOP_TO_BOTTOM:
            if row + size > 10:
                return False
            left_col_pieces = self.col_pieces[col]
            for i in range(size):
                val = self.get_value(row + i, col)
                if self.row_pieces[row + i] == 0:
                    if val not in HintPosition and val is not BoardPosition.PROBABLY_BOAT:
                        return False
                left_col_pieces += 1 if val in HintPosition or val is BoardPosition.PROBABLY_BOAT else 0
            if left_col_pieces < size:
                return False
            return True
        raise NotImplementedError

    def detect_boats(self) -> (int, int, int, int):
        """Detecta barcos no tabuleiro já colocados pelas dicas."""
        ones = twos = thres = fours = 0

        # como o for loop vai de cima para baixo da esquerda para direita,
        # validaremos somente as posições top, left e center, para nao repetir a contagem
        for row in range(0, 10):
            for col in range(0, 10):
                size = 0
                value = self.get_value(row, col)
                if value not in (HintPosition.WATER, BoardPosition.WATER, BoardPosition.UNKNOWN):
                    if value is HintPosition.CENTER:
                        size = 1
                        self.already_placed_boats.append((row, col, size, PlaceDirection.LEFT_TO_RIGHT))
                    if value is HintPosition.TOP:
                        for i in range(4):
                            size += 1
                            if not is_board_position(self.get_value(row + i, col)):
                                if not self.get_value(row + i - 1, col) is HintPosition.BOTTOM:
                                    size = 0
                                else:
                                    size -= 1
                                break
                        if size > 0:
                            self.already_placed_boats.append((row, col, size, PlaceDirection.TOP_TO_BOTTOM))
                    if value is HintPosition.LEFT:
                        for i in range(4):
                            size += 1
                            if not is_board_position(self.get_value(row, col + i)):
                                if not self.get_value(row, col + i - 1) is HintPosition.RIGHT:
                                    size = 0
                                break
                        if size > 0:
                            self.already_placed_boats.append((row, col, size, PlaceDirection.LEFT_TO_RIGHT))
                if size == 1:
                    ones += 1
                elif size == 2:
                    twos += 1
                elif size == 3:
                    thres += 1
                elif size == 4:
                    fours += 1

        return (ones, twos, thres, fours)

    @staticmethod
    def parse_pieces() -> (np.array, np.array):
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board."""
        line = sys.stdin.readline().split()
        row_pieces = np.array([int(x) for x in line if x != "ROW"])

        line = sys.stdin.readline().split()
        col_pieces = np.array([int(x) for x in line if x != "COLUMN"])

        return (row_pieces, col_pieces)

    def fill_empty_spaces(self):
        for row in range(10):
            for col in range(10):
                if self.row_pieces[row] == 0 or self.col_pieces[col] == 0:
                    if self.get_value(row, col) is BoardPosition.UNKNOWN:
                        self.set_value(row, col, BoardPosition.WATER)

    def print(self):
        """Imprime o tabuleiro."""
        string = " " if DEBUG else ""
        if DEBUG:
            print("  " + string.join([str(x) for x in self.col_pieces]))
        for row in range(len(self.positions)):
            val = self.positions[row]
            if DEBUG:
                print(str(self.row_pieces[row]) + " ", end="")
            print(string.join([str(i.value) for i in val]))

    def can_place_boat(self, row: int, col: int, size: int, direction: PlaceDirection):
        """Verifica se é possível inserir um barco de tamanho 'size' na posição
        ('row', 'col') com a orientação 'direction'."""

        if (row, col, size, direction) in self.already_placed_boats:
            return False

        boat_positions = []
        adjacent_positions = []
        if not self.have_enought_left_pieces_for_boat(row, col, size, direction):
            return False
        if direction is PlaceDirection.LEFT_TO_RIGHT:
            adjacent_positions.append(self.get_value(row, col - 1))
            adjacent_positions.append(self.get_value(row, col + size))
            for i in range(size):
                boat_positions.append(self.get_value(row, col + i))
                adjacent_positions.extend(self.adjacent_vertical_values(row, col + i))
                adjacent_positions.extend(self.adjacent_diagonal_values(row, col + i))
        if direction is PlaceDirection.RIGHT_TO_LEFT:
            raise NotImplementedError
        if direction is PlaceDirection.TOP_TO_BOTTOM:
            adjacent_positions.append(self.get_value(row - 1, col))
            adjacent_positions.append(self.get_value(row + size, col))
            for i in range(size):
                boat_positions.append(self.get_value(row + i, col))
                adjacent_positions.extend(self.adjacent_horizontal_values(row + i, col))
                adjacent_positions.extend(self.adjacent_diagonal_values(row + i, col))
        if direction is PlaceDirection.BOTTOM_TO_TOP:
            raise NotImplementedError

        can_place = True
        for i in range(len(boat_positions)):
            if not can_place:
                return False
            val = boat_positions[i]
            if val in HintPosition:
                is_first = i == 0
                is_last = i == len(boat_positions) - 1
                if direction is PlaceDirection.LEFT_TO_RIGHT:
                    if is_first:
                        can_place = val is HintPosition.LEFT
                    elif is_last:
                        can_place = val is HintPosition.RIGHT
                    else:
                        can_place = val is HintPosition.MIDDLE
                elif direction is PlaceDirection.TOP_TO_BOTTOM:
                    if is_first:
                        can_place = val is HintPosition.TOP
                    elif is_last:
                        can_place = val is HintPosition.BOTTOM
                    else:
                        can_place = val is HintPosition.MIDDLE
            else:
                can_place = is_placeble_position(val)
        if not can_place:
            return False

        is_surrounded_by_water = all([is_empty_position(i) for i in adjacent_positions])
        return can_place and is_surrounded_by_water


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []

        next_size = state.board.remaining_boats.get_next_size()

        if next_size == 0:
            return actions

        if next_size == 1:
            for row in range(10):
                for col in range(10):
                    if state.board.can_place_boat(row, col, 1, PlaceDirection.LEFT_TO_RIGHT):
                        actions.append((row, col, 1, PlaceDirection.LEFT_TO_RIGHT))
        else:
            for row in range(10):
                for col in range(10):
                    if state.board.can_place_boat(row, col, next_size, PlaceDirection.LEFT_TO_RIGHT):
                        actions.append((row, col, next_size, PlaceDirection.LEFT_TO_RIGHT))
                    if state.board.can_place_boat(row, col, next_size, PlaceDirection.TOP_TO_BOTTOM):
                        actions.append((row, col, next_size, PlaceDirection.TOP_TO_BOTTOM))

        return actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        row, col, size, direction = action

        new_board = state.board.copy()
        new_board.place_boat_and_waters(row, col, size, direction)
        new_board.remaining_boats.decrease_boat_count(size)
        new_board.fill_empty_spaces()

        return BimaruState(new_board)

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        has_remaining_boats = any([val != 0 for val in state.board.remaining_boats.get_values()])
        has_invalid_row = np.any(state.board.row_pieces != 0)
        has_invalid_col = np.any(state.board.col_pieces != 0)

        return not has_remaining_boats and not has_invalid_row and not has_invalid_col

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        sum_of_remaining_pieces = np.sum(node.state.board.row_pieces + node.state.board.col_pieces)
        ones, twos, threes, fours = node.state.board.remaining_boats.get_values()
        remaining_boats = ones + 4 * twos + 6 * threes + 8 * fours
        value = 10 * sum_of_remaining_pieces + 20 * remaining_boats
        value += 100 * node.state.board.incomplete_boats()
        return value


if __name__ == "__main__":
    board = Board.parse_instance()
    problem = Bimaru(board)
    goal_node = depth_first_tree_search(problem)
    goal_node.state.board.print()
