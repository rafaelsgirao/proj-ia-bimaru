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

    def adjascent_horizontal_values(self, row, col):
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
                    print(".", end="")
                elif value == 1:
                    left, right = self.adjascent_horizontal_values(row, col)
                    top, bottom = self.adjacent_vertical_values(row, col)
                    if left == 1 and right == 0:
                        print("r", end="")
                    elif left == 1 and right == 1:
                        print("m", end="")
                    elif left == 0 and right == 1:
                        print("l", end="")
                    elif top == 1 and bottom == 0:
                        print("b", end="")
                    elif top == 1 and bottom == 1:
                        print("m", end="")
                    elif top == 0 and bottom == 1:
                        print("t", end="")
                    elif top == 0 and bottom == 0 and left == 0 and right == 0:
                        print("c", end="")
            print()

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe board."""
        board = Board()

        board.positions = np.full((10, 10), 0)

        # read rows
        line = stdin.readline().split()
        board.rows = np.array([int(x) for x in line if x != "ROW"])

        # read columns
        line = stdin.readline().split()
        board.cols = np.array([int(x) for x in line if x != "COLUMN"])

        return board


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        self.parse_hints()
        # TODO

    def parse_hints(self):
        # Numpy array
        self.hint_positions = np.full((10 + 2, 10 + 2), 1)
        # add hints
        hint_total = int(stdin.readline())
        for _ in range(hint_total):
            line = stdin.readline().split()
            row = int(line[1]) + 1
            col = int(line[2]) + 1
            value = line[3]

            if value == "C":
                self.hint_positions[row - 1 : row + 2, col - 1 : col + 2] = 0
                self.hint_positions[row, col] = 1
            elif value == "B":
                self.hint_positions[row - 2 : row + 1, col - 1] = 0
                self.hint_positions[row - 2 : row + 1, col + 1] = 0
                self.hint_positions[row + 1, col - 1 : col + 2] = 0
                self.hint_positions[row, col] = 1
            elif value == "T":
                self.hint_positions[row : row + 2, col - 1] = 0
                self.hint_positions[row : row + 2, col + 1] = 0
                self.hint_positions[row - 1, col - 1 : col + 1] = 0
                self.hint_positions[row, col] = 1
            elif value == "L":
                self.hint_positions[row - 1 : row + 2, col - 1] = 0
                self.hint_positions[row + 1, col - 1 : col + 3] = 0
                self.hint_positions[row - 1, col - 1 : col + 3] = 0
                self.hint_positions[row, col] = 1
            elif value == "R":
                self.hint_positions[row - 1 : row + 2, col + 1] = 0
                self.hint_positions[row + 1, col - 2 : col + 1] = 0
                self.hint_positions[row - 1, col - 2 : col + 1] = 0
                self.hint_positions[row, col] = 1
            elif value == "W":
                self.hint_positions[row, col] = 0

            elif value == "M":
                self.hint_positions[row - 1, col - 1] = 0
                self.hint_positions[row - 1, col + 1] = 0

                self.hint_positions[row + 1, col - 1] = 0
                self.hint_positions[row + 1, col + 1] = 0
        # Remove padding
        self.hint_positions = self.hint_positions[1:-1, 1:-1]

    possible_ships = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]

    possible_matrices = {}

    def gen_matrices_1(self):
        m = []
        dumb = np.zeros((10, 10))
        for (row, col), _ in np.ndenumerate(dumb):
            print((row, col))
            #  row, col = *index
            new_matrix = np.zeros((10, 10))
            new_matrix[row, col] = 1
            m.append(new_matrix)
        return m

    def has_conflicts_with_hints(self, n):
        # Todo
        pass

    def gen_matrices(self, n):
        m = []

        for (row, col), _ in np.ndenumerate(np.zeros((10, 10 - n + 1))):
            new_matrix = np.zeros((10, 10))
            for i in range(n):
                new_matrix[row, col + i] = 1
                # if self.problem.insertion_conflicts_with_hints(row, col + i):
                # should_continue = True
                # break
            # if should_continue:
            # continue
            m.append(new_matrix)

        another_m = [mx.transpose() for mx in m] if n > 1 else []

        another_m.extend(m)

        final = []

        for matrix in another_m:
            if not self.matrix_conflicts_with_hints(matrix):
                final.append(matrix)

        return final

    def gen_possible_matrices(self):
        self.possible_matrices[1] = self.gen_matrices_1()
        self.possible_matrices[2] = self.gen_matrices(2)
        self.possible_matrices[3] = self.gen_matrices(3)
        self.possible_matrices[4] = self.gen_matrices(4)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

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

        # Only check if all rows and columns are filled:
        # More profound checks should be done before/when filling positions.
        return np.all(state.problem.cols == 0) and np.all(state.problem.rows == 0)

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
    # print(self.hint_positions)
    # print(self.hint_positions.shape)
    board = Board.parse_instance()
    problem = Bimaru(board)
    board.print()
    # print(problem.gen_matrices(1))
    # initial_state = BimaruState(board)

    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
