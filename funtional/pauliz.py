from ..macro import C_DTYPE
from .gate_wrapper import gate_wrapper
import torch


_z_mat_dict = {
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

def pauliz(
   q_device,
   wires,
   params=None,
   n_wires=None,
   static=False,
   parent_graph=None,
   inverse=False,
   comp_method="bmm",
):
   """Pauli Z 게이트 연산 수행
   
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
   name = "pauliz"
   mat = _z_mat_dict[name]
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

def cz(
   q_device,
   wires,
   params=None,
   n_wires=None,
   static=False,
   parent_graph=None,
   inverse=False,
   comp_method="bmm",
):
   """제어된 Z 게이트 연산 수행
   
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
   name = "cz"
   mat = _z_mat_dict[name]
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
z = pauliz