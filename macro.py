# 전역적으로 사용되는 기본 상수와 설정들을 정의

import math
import torch
import numpy as np

from string import ascii_lowercase # 알파벳 소문자 

C_DTYPE = torch.complex64  # 복소수 데이터 타입 64비트
F_DTYPE = torch.float32 # 실수 데이터 타입 32비트

ABC = ascii_lowercase # 알파벳 소문자
ABC_ARRAY = np.array(list(ABC)) # 알파벳 소문자 배열

INV_SQRT2 = 1 / math.sqrt(2)

print(ABC_ARRAY)