import functools
import torch
import numpy as np

from typing import Callable
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY



def apply_unitary_einsum(state, mat, wires):
   """einsum을 사용하여 상태벡터에 유니타리 연산을 적용

   Args:
       state (torch.Tensor): 상태벡터
       mat (torch.Tensor): 게이트의 유니타리 행렬  
       wires (int or List[int]): 연산을 적용할 큐비트 인덱스

   Returns:
       torch.Tensor: 연산이 적용된 새로운 상태벡터
   """
   device_wires = wires

   # 배치를 제외한 전체 와이어(큐비트) 수
   total_wires = len(state.shape) - 1

   # 행렬이 배치를 포함하는지 확인
   if len(mat.shape) > 2:
       is_batch_unitary = True
       bsz = mat.shape[0]
       shape_extension = [bsz]
   else:
       is_batch_unitary = False
       shape_extension = []

   # 행렬을 적절한 형태로 변환
   mat = mat.view(shape_extension + [2] * len(device_wires) * 2)
   mat = mat.type(C_DTYPE).to(state.device)

   # 양자 상태의 텐서 인덱스 설정
   state_indices = ABC[:total_wires]

   # 이 연산의 영향을 받는 상태 인덱스
   affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())

   # 새로운 인덱스 생성
   new_indices = ABC[total_wires: total_wires + len(device_wires)]

   # 상태의 새 인덱스 생성 (영향받은 인덱스를 새 인덱스로 대체)
   new_state_indices = functools.reduce(
       lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
       zip(affected_indices, new_indices),
       state_indices,
   )

   # 배치 차원을 위한 인덱스 처리
   state_indices = ABC[-1] + state_indices
   new_state_indices = ABC[-1] + new_state_indices
   if is_batch_unitary:
       new_indices = ABC[-1] + new_indices

   # einsum 표기법으로 인덱스 조합
   einsum_indices = (
       f"{new_indices}{affected_indices}," 
       f"{state_indices}->{new_state_indices}"
   )

   # einsum을 사용하여 새로운 상태 계산
   new_state = torch.einsum(einsum_indices, mat, state)

   return new_state

def apply_unitary_bmm(state, mat, wires):
   """batch matrix multiplication을 사용하여 상태벡터에 유니타리 연산을 적용

   Args:
       state (torch.Tensor): 상태벡터
       mat (torch.Tensor): 게이트의 유니타리 행렬
       wires (int or List[int]): 연산을 적용할 큐비트 인덱스

   Returns:
       torch.Tensor: 연산이 적용된 새로운 상태벡터
   """
   device_wires = wires

   # 행렬을 복소수 타입으로 변환하고 상태벡터와 같은 디바이스로 이동
   mat = mat.type(C_DTYPE).to(state.device)

   # 타겟 큐비트의 차원을 조정 
   devices_dims = [w + 1 for w in device_wires]
   permute_to = list(range(state.dim()))
   for d in sorted(devices_dims, reverse=True):
       del permute_to[d]
   permute_to = permute_to[:1] + devices_dims + permute_to[1:]
   permute_back = list(np.argsort(permute_to))
   original_shape = state.shape
   
   # 상태를 행렬 곱셈에 적합한 형태로 변환
   permuted = state.permute(permute_to).reshape([original_shape[0], mat.shape[-1], -1])

   if len(mat.shape) > 2:
       # 행렬과 상태 모두 배치 모드인 경우
       new_state = mat.bmm(permuted)
   else:
       # 행렬은 배치가 없고 상태만 배치 모드인 경우
       bsz = permuted.shape[0]
       expand_shape = [bsz] + list(mat.shape)
       new_state = mat.expand(expand_shape).bmm(permuted)

   # 결과를 원래 형태로 복원
   new_state = new_state.view(original_shape).permute(permute_back)

   return new_state


def gate_wrapper(
        name,           # 게이트 이름
        mat,           # 게이트의 유니타리 행렬
        method,        # 'bmm' 또는 'einsum' 행렬 곱셈 방법
        q_device,      # 양자 상태를 저장하는 디바이스
        wires,         # 게이트를 적용할 큐비트
        params=None,   # 게이트 파라미터
        n_wires=None,  # 큐비트 수
        inverse=False, # 역연산 여부
):
    # 파라미터 처리
    if params is not None:
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=F_DTYPE)
        if params.dim() == 1:
            params = params.unsqueeze(-1)
        elif params.dim() == 0:
            params = params.unsqueeze(-1).unsqueeze(-1)

    # 와이어 처리
    wires = [wires] if isinstance(wires, int) else wires

    # 행렬 계산
    if isinstance(mat, Callable):
        matrix = mat(params)
    else:
        matrix = mat

    # 역연산 처리
    if inverse:
        matrix = matrix.conj()
        if matrix.dim() == 3:
            matrix = matrix.permute(0, 2, 1)
        else:
            matrix = matrix.permute(1, 0)

    # 상태 업데이트
    state = q_device.states
    if method == "einsum":
        q_device.states = apply_unitary_einsum(state, matrix, wires)
    elif method == "bmm":
        q_device.states = apply_unitary_bmm(state, matrix, wires)