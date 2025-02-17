from ..macro import C_DTYPE
from .gate_wrapper import gate_wrapper
import torch


def multicnot_matrix(n_wires):
   """n개의 큐비트에 대한 CNOT 행렬 계산
   
   Args:
       n_wires (int): 큐비트 수
       
   Returns:
       torch.Tensor: 계산된 유니타리 행렬
   """
   mat = torch.eye(2**n_wires, dtype=C_DTYPE)
   mat[-1][-1] = 0
   mat[-2][-2] = 0  
   mat[-1][-2] = 1
   mat[-2][-1] = 1
   return mat

def multixcnot_matrix(n_wires):
   """n개의 큐비트에 대한 XCNOT 행렬 계산
   
   Args:
       n_wires (int): 큐비트 수
       
   Returns:
       torch.Tensor: 계산된 유니타리 행렬
   """
   mat = torch.eye(2**n_wires, dtype=C_DTYPE)
   mat[0][0] = 0
   mat[1][1] = 0
   mat[0][1] = 1
   mat[1][0] = 1
   return mat

# 게이트들의 행렬 정의
_x_mat_dict = {
   "paulix": torch.tensor([[0, 1], [1, 0]], dtype=C_DTYPE),
   "cnot": torch.tensor(
       [[1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1], 
        [0, 0, 1, 0]], dtype=C_DTYPE
   ),
   "multicnot": multicnot_matrix,
   "multixcnot": multixcnot_matrix,
}

def paulix(
   q_device,
   wires,
   params=None,
   n_wires=None,
   static=False,
   parent_graph=None,
   inverse=False,
   comp_method="bmm",
):
   """Pauli X 게이트 연산 수행
   
   Args:
       q_device: 양자 상태를 저장하는 디바이스
       wires: 게이트를 적용할 큐비트
       params: 게이트 파라미터 (기본값: None)
       n_wires: 게이트를 적용할 큐비트 수 (기본값: None)
       static: 정적 모드 사용 여부 (기본값: False)
       parent_graph: 부모 양자 그래프 (기본값: None)
       inverse: 역연산 여부 (기본값: False)
       comp_method: 행렬-벡터 곱셈 방법 ('bmm' 또는 'einsum') (기본값: 'bmm')
   """
   name = "paulix"
   mat = _x_mat_dict[name]
   gate_wrapper(
       name=name,
       mat=mat,
       method=comp_method,
       q_device=q_device,
       wires=wires,
       params=params,
       n_wires=n_wires,
       static=static,
       parent_graph=parent_graph,
       inverse=inverse,
   )

def cnot(
   q_device,
   wires,
   params=None,
   n_wires=None,
   static=False,
   parent_graph=None,
   inverse=False,
   comp_method="bmm",
):
   """CNOT 게이트 연산 수행
   
   Args:
       q_device: 양자 상태를 저장하는 디바이스
       wires: 게이트를 적용할 큐비트
       params: 게이트 파라미터 (기본값: None)
       n_wires: 게이트를 적용할 큐비트 수 (기본값: None)
       static: 정적 모드 사용 여부 (기본값: False)
       parent_graph: 부모 양자 그래프 (기본값: None)
       inverse: 역연산 여부 (기본값: False)
       comp_method: 행렬-벡터 곱셈 방법 ('bmm' 또는 'einsum') (기본값: 'bmm')
   """
   name = "cnot"
   mat = _x_mat_dict[name]
   gate_wrapper(
       name=name,
       mat=mat,
       method=comp_method,
       q_device=q_device,
       wires=wires,
       params=params,
       n_wires=n_wires,
       static=static,
       parent_graph=parent_graph,
       inverse=inverse,
   )

# 별칭 정의
x = paulix
cx = cnot