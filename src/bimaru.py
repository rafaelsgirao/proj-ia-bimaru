# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 32:
# 99309 Rafael Girão
# 104147 Guilherme Marcondes


import numpy as np
from sys import stdin, exit

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

BOARD_SIZE = (10, 10)

DEBUG = True


# TODO: No final do projeto, remover esta função e ctrl+f por debug
def debug(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def rprint(msg):
    print(msg, end="")


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

    def adjacent_vertical_values(self, row, col):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        if row == 0:
            return [0, self.positions[row + 1, col]]
        elif row == 9:
            return [self.positions[row - 1, col], 0]
        else:
            return [self.positions[row - 1, col], self.positions[row + 1, col]]

    def adjacent_horizontal_values(self, row, col):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if col == 0:
            return [0, self.positions[row, col + 1]]
        elif col == 9:
            return [self.positions[row, col - 1], 0]
        else:
            return [self.positions[row, col - 1], self.positions[row, col + 1]]

    def print(self):
        """Imprime o tabuleiro."""
        # todo: consider hints
        for row in range(10):
            for col in range(10):
                value = self.positions[row, col]
                if value == 0:
                    rprint(".")
                elif value == 1:
                    left, right = self.adjacent_horizontal_values(row, col)
                    top, bottom = self.adjacent_vertical_values(row, col)
                    if left == 1 and right == 0:
                        rprint("r")
                    elif left == 1 and right == 1:
                        rprint("m")
                    elif left == 0 and right == 1:
                        rprint("l")
                    elif top == 1 and bottom == 0:
                        rprint("b")
                    elif top == 1 and bottom == 1:
                        rprint("m")
                    elif top == 0 and bottom == 1:
                        rprint("t")
                    elif top == 0 and bottom == 0 and left == 0 and right == 0:
                        rprint("c")
            print()

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe board."""
        board = Board()

        board.remaining_ships = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
        board.used_ships = []  # A list of tuples, e.g [(ship_size, index)]
        board.positions = np.full((10, 10), 0)

        return board

    def copy(self):
        """Retorna uma instância de Board idêntica à atual."""
        new_board = Board()
        new_board.remaining_ships = self.remaining_ships.copy()
        new_board.positions = self.positions.copy()
        new_board.used_ships = board.used_ships.copy()

        return new_board


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        self.parse_limits()
        self.parse_hints()
        self.gen_possible_matrices()
        # TODO

    def parse_limits(self):
        # read rows
        line = stdin.readline().split()
        self.rows = np.array([int(x) for x in line if x != "ROW"])

        # read columns
        line = stdin.readline().split()
        self.cols = np.array([int(x) for x in line if x != "COLUMN"])

    def parse_hints(self):
        # Numpy array
        self.hints = np.full((10 + 2, 10 + 2), 1)
        # add hints
        hint_total = int(stdin.readline())
        for _ in range(hint_total):
            line = stdin.readline().split()
            row = int(line[1]) + 1
            col = int(line[2]) + 1
            value = line[3]

            if value == "C":
                self.hints[row - 1 : row + 2, col - 1 : col + 2] = 0
                self.hints[row, col] = 1
            elif value == "B":
                self.hints[row - 2 : row + 1, col - 1] = 0
                self.hints[row - 2 : row + 1, col + 1] = 0
                self.hints[row + 1, col - 1 : col + 2] = 0
                self.hints[row, col] = 1
            elif value == "T":
                self.hints[row : row + 2, col - 1] = 0
                self.hints[row : row + 2, col + 1] = 0
                self.hints[row - 1, col - 1 : col + 1] = 0
                self.hints[row, col] = 1
            elif value == "L":
                self.hints[row - 1 : row + 2, col - 1] = 0
                self.hints[row + 1, col - 1 : col + 3] = 0
                self.hints[row - 1, col - 1 : col + 3] = 0
                self.hints[row, col] = 1
            elif value == "R":
                self.hints[row - 1 : row + 2, col + 1] = 0
                self.hints[row + 1, col - 2 : col + 1] = 0
                self.hints[row - 1, col - 2 : col + 1] = 0
                self.hints[row, col] = 1
            elif value == "W":
                self.hints[row, col] = 0

            elif value == "M":
                self.hints[row - 1, col - 1] = 0
                self.hints[row - 1, col + 1] = 0

                self.hints[row + 1, col - 1] = 0
                self.hints[row + 1, col + 1] = 0
        # Remove padding
        self.hints = self.hints[1:-1, 1:-1]

    def has_conflicts_with_hints(self, matrix):
        debug(f"\nMATRIZ DE INPUT: \n {matrix}")
        debug(f"\nMATRIZ DAS HINTS: \n {self.hints}")
        debug("matrix - self.hints:")
        result = self.hints - matrix
        debug(f"\n {result}")
        return np.any(result < 0)

    possible_matrices = {}

    def gen_matrices(self, n):
        m = []

        for (row, col), _ in np.ndenumerate(np.zeros((10, 10 - n + 1))):
            new_matrix = np.zeros((10, 10), dtype=int)
            for i in range(n):
                new_matrix[row, col + i] = 1
            m.append(new_matrix)

        another_m = [mx.transpose() for mx in m] if n > 1 else []

        another_m.extend(m)

        final = []

        for matrix in another_m:
            if not self.has_conflicts_with_hints(matrix):
                final.append(matrix)

        return final

    def gen_possible_matrices(self):
        for i in (1, 2, 3, 4):
            self.possible_matrices[i] = self.gen_matrices(i)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        rem_ships = state.board.remaining_ships.copy()
        if len(rem_ships) == 0:
            yield (None, None)

            # Get the next largest ship that hasn't been placed.
            # while True:  # Look breaks when rem_ships.pop fails.
        next_ship = rem_ships.pop(0)
        debug(f"next_ship = {next_ship}")
        for matrix_i in range(len(self.possible_matrices[next_ship])):
            m = self.possible_matrices[next_ship][matrix_i]
            if not self.matrices_in_conflict(m, state.board.positions):
                yield (next_ship, matrix_i)
        raise ValueError  # FIXME: retornar tuplo vazio p/ nao haver acoes

    """
    Verifica se dois boards estão em conflito ou se a sua junção pode existir. (TODO: verificar board resultante c/ is_valid_board)
    """

    @staticmethod
    def matrices_in_conflict(m1, m2) -> bool:
        #    m1, m2 = b1.positions, b2.positions
        m_sums = np.empty_like(m1)
        m2_pad = np.pad(m2, pad_width=1, mode="constant", constant_values=0)
        #  print(m2_pad)
        m_filter = np.array(([1, 1, 1], [1, 0, 1], [1, 1, 1]))
        for (i, j), _ in np.ndenumerate(m_sums):
            # m1_around_point = m1[i: i+ 2, j: j+2]
            m2_around_point = m2_pad[i : i + 3, j : j + 3]  # 3x3 matrix around point
            # debug(f"({i}, {j})")
            # debug(m2_around_point)
            # debug("---")
            m_sums[i, j] = np.sum(m2_around_point * m_filter) * m1[i, j] + m1[i, j]

        return np.any(m_sums > 1)

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        debug(f"action = {action}")
        new_board = state.board.copy()
        ship_matrix = self.possible_matrices[action[0]][action[1]]
        new_board.positions += ship_matrix[1]
        # debug(new_board.remaining_ships)
        new_board.remaining_ships.remove(action[0])
        new_board.used_ships.append(action)

        return BimaruState(new_board)

    def is_valid_board(self, board, goal_test=False):
        def cmp(a, b, goal_test):
            if goal_test:
                return a == b
            else:
                return a > b

        for row in range(len(self.rows)):
            board_row_sum = np.sum(board.positions[row, :])
            # debug(
            #    f"board_row_sum ({row}) = {board_row_sum} | row_val = {self.rows[row]}"
            # )
            #    if board_row_sum > self.rows[row]:
            if cmp(board_row_sum, self.rows[row]):
                return False
        for col in range(len(self.cols)):
            board_col_sum = np.sum(board.positions[:, col])
            # debug(
            #    f"board_col_sum ({col}) = {board_col_sum} | col_val = {self.cols[row]}"
            # )
            #  if board_col_sum > self.cols[col]:
            if cmp(board_col_sum, self.cols[col]):
                return False
        return True

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""

        # Only check if all rows and columns are filled:
        # More profound checks should be done before/when filling positions.
        # return np.all(state.problem.cols == 0) and np.all(state.problem.rows == 0)
        return self.is_valid_board(state.board, goal_test=True)

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    @classmethod
    def has_conflicts_in_matrix(cls, b1, b2) -> bool:
        for r in b1:
            for c in r:
                if cls.has_conflicts_in_position((r, c), b1, b2):
                    return True
        return False

    @staticmethod
    def has_conflicts_in_position(pos, b1, b2) -> bool:
        row, col = pos

        s = b1[row, col]
        s += b2[row - 1 : row + 1, col - 1 : col + 1]

        return s <= 1


if __name__ == "__main__":
    # print(self.hints)
    # print(self.hints.shape)
    board = Board.parse_instance()
    problem = Bimaru(board)
    board.print()
    # print(problem.gen_matrices(1))
    s1 = BimaruState(board)

    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
