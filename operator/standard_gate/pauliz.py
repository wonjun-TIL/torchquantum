class PauliZ(Observable, metaclass=ABCMeta):
   """Class for Pauli Z Gate."""
   
   num_params = 0        # 파라미터 없음
   num_wires = 1        # 단일 큐비트 게이트
   eigvals = torch.tensor([1, -1], dtype=C_DTYPE)  # 고유값
   op_name = "pauliz"   # 게이트 이름
   matrix = mat_dict["pauliz"]  # 게이트 행렬
   func = staticmethod(tqf.pauliz)  # 실제 연산 함수

   @classmethod
   def _matrix(cls, params):
       return cls.matrix

   @classmethod
   def _eigvals(cls, params):
       return cls.eigvals

   def diagonalizing_gates(self):
       return []  # Z 게이트는 이미 대각화되어 있음