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

    class Position:
        """Representação interna de uma posição do tabuleiro."""

        def __init__(self, value: str, hint=False) -> None:
            self.value = value
            self.hint = hint

        def __str__(self) -> str:
            return self.value.upper() if self.hint else self.value

        def __repr__(self) -> str:
            return self.__str__()

    # Singleton.
    class Empty(Position):
        instance = None

        def __init__(self):
            super().__init__(".")

            if Board.Empty.instance:
                return
            Board.Empty.instance = self
            Board.Empty.__new__ = lambda _: Board.Empty.instance

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.hint_positions[row][col].value

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):  # type: ignore
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        return (self.hint_positions[row - 1][col].value, self.hint_positions[row + 1][col].value)

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):  # type: ignore
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.hint_positions[row][col - 1].value, self.hint_positions[row][col + 1].value)

    def diagonal_values(self, row: int, col: int) -> (str, str, str, str):  # type: ignore
        """Devolve os valores nas quatro posições diagonais,
        respectivamente."""
        return (
            self.hint_positions[row - 1][col - 1].value,
            self.hint_positions[row - 1][col + 1].value,
            self.hint_positions[row + 1][col - 1].value,
            self.hint_positions[row + 1][col + 1].value,
        )

    def print(self):
        for row in range(len(self.positions)):
            for col in range(len(self.positions[row])):
                print(self.positions[row][col], end=" ")
            print()

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board."""
        board = Board()

        # Numpy array
        board.hint_positions = np.full((10, 10), Board.Empty())
        board.positions = np.full((10, 10), 0)

        # read rows
        line = stdin.readline().split()
        board.rows = np.array([int(x) for x in line if x != "ROW"])

        # read columns
        line = stdin.readline().split()
        board.cols = np.array([int(x) for x in line if x != "COLUMN"])

        # add hints
        hint_total = int(stdin.readline())
        for _ in range(hint_total):
            line = stdin.readline().split()
            row = int(line[1])
            col = int(line[2])
            value = line[3]
            board.hint_positions[row][col] = Board.Position(value, hint=True)

        return board

    def matrix_conflicts_with_hints(self, matrix) -> bool:
        """Verifica se uma matriz de posições do tabuleiro é válida, isto é,
        se não viola as restrições impostas pelos valores das pistas."""
        
        boat = []
        for (row, col), _ in np.ndenumerate(matrix):
            if matrix[row][col] == 1:
                boat.append((row, col))
            if len(boat) == 4:
                break
        
        boat_size = len(boat)

        if boat_size == 1:
            row, col = boat[0]
            adjacent_values = self.adjacent_vertical_values(row, col) + self.adjacent_horizontal_values(row, col) + self.diagonal_values(row, col)
            if not all(value == Board.Empty() for value in adjacent_values):
                return False
            return self.hint_positions[row][col] == Board.Empty() or self.hint_positions[row][col].value == "C"

        return True
            


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        # TODO

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

    def gen_matrices(self, n):
        m = []

        for (row, col), _ in np.ndenumerate(np.zeros((10, 10 - n + 1))):
            new_matrix = np.zeros((10, 10))
            for i in range(n):
                new_matrix[row, col + i] = 1
                # if self.board.insertion_conflicts_with_hints(row, col + i):
                    #should_continue = True
                    #break
            #if should_continue:
                #continue
            m.append(new_matrix)

        another_m = [mx.transpose() for mx in m] if n > 1 else []

        another_m.extend(m)

        final = []

        for matrix in another_m:
            if self.board.matrix_conflicts_with_hints(matrix):
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
        return np.all(state.board.cols == 0) and np.all(state.board.rows == 0)

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
    board = Board.parse_instance()
    board.print()
    problem = Bimaru(board)
    print(problem.gen_matrices(1))
    initial_state = BimaruState(board)

    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
