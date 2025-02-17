class PauliX(Observable, metaclass=ABCMeta):
    """Class for Pauli X Gate."""

    num_params = 0        # 파라미터 없음
    num_wires = 1        # 단일 큐비트 게이트
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)  # 고유값
    op_name = "paulix"   # 게이트 이름
    matrix = mat_dict["paulix"]  # 게이트 행렬
    func = staticmethod(tqf.paulix)  # 실제 연산 함수

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return [tq.Hadamard()]