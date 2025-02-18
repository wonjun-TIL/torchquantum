import torch
import numpy as np


from ..macro import C_DTYPE


def rx_matrix(params: torch.Tensor) -> torch.Tensor:
    """RX 게이트의 유니타리 행렬을 계산
    
    Args:
        params (torch.Tensor): 회전각
        
    Returns:
        torch.Tensor: 2x2 유니타리 행렬
    """
    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    jsi = 1j * torch.sin(-theta / 2)
    
    return torch.stack(
        [torch.cat([co, jsi], dim=-1), 
         torch.cat([jsi, co], dim=-1)], 
        dim=-2
    ).squeeze(0)




def rx(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """RX 게이트 연산 수행
    
    Args:
        q_device: 양자 상태를 저장하고 있는 디바이스
        wires: 게이트를 적용할 큐비트
        params: 회전각 파라미터
        comp_method: 행렬-벡터 곱셈 방법 ('bmm' 또는 'einsum')
    """
    name = "rx"
    mat = _rx_mat_dict[name]
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