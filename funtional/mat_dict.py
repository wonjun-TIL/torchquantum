import torch
from macro import C_DTYPE

mat_dict = {
    # Pauli-X 게이트와 관련 게이트들
    "paulix": torch.tensor([[0, 1], [1, 0]], dtype=C_DTYPE),
    "cnot": torch.tensor(
        [[1, 0, 0, 0], 
         [0, 1, 0, 0], 
         [0, 0, 0, 1], 
         [0, 0, 1, 0]], dtype=C_DTYPE
    ),
    "multicnot": multicnot_matrix,
    "multixcnot": multixcnot_matrix,

    # Pauli-Y 게이트와 관련 게이트들
    "pauliy": torch.tensor([[0, -1j], [1j, 0]], dtype=C_DTYPE),
    "cy": torch.tensor(
        [[1, 0, 0, 0], 
         [0, 1, 0, 0], 
         [0, 0, 0, -1j], 
         [0, 0, 1j, 0]], dtype=C_DTYPE
    ),

    # Pauli-Z 게이트와 관련 게이트들
    "pauliz": torch.tensor([[1, 0], [0, -1]], dtype=C_DTYPE),
    "cz": torch.tensor(
        [[1, 0, 0, 0], 
         [0, 1, 0, 0], 
         [0, 0, 1, 0], 
         [0, 0, 0, -1]], dtype=C_DTYPE
    ),
    "ccz": torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, -1]], dtype=C_DTYPE
    ),
}