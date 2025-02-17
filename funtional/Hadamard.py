_hadamard_mat_dict = {
   # 기본 Hadamard 게이트
   "hadamard": torch.tensor(
       [[INV_SQRT2, INV_SQRT2], 
        [INV_SQRT2, -INV_SQRT2]], dtype=C_DTYPE
   ),
   # Special Hadamard 게이트 (π/8 회전)
   "shadamard": torch.tensor(
       [[np.cos(np.pi / 8), -np.sin(np.pi / 8)],
        [np.sin(np.pi / 8), np.cos(np.pi / 8)]], 
       dtype=C_DTYPE
   ),
   # 제어 Hadamard 게이트
   "chadamard": torch.tensor(
       [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, INV_SQRT2, INV_SQRT2],
        [0, 0, INV_SQRT2, -INV_SQRT2]], 
       dtype=C_DTYPE
   ),
}

def hadamard(
   q_device: QuantumDevice,
   wires: Union[List[int], int],
   params: torch.Tensor = None,
   n_wires: int = None,
   static: bool = False,
   parent_graph=None,
   inverse: bool = False,
   comp_method: str = "bmm",
):
   """Hadamard 게이트 연산 수행
   
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
   name = "hadamard"
   mat = _hadamard_mat_dict[name]
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
h = hadamard
ch = chadamard
sh = shadamard