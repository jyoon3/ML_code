# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 12:28:27 2025

@author: jy129
"""

import openvino as ov
from openvino.runtime import Core
import sys

try:
    # Core 객체 생성 시도
    ie = Core()
    print("🎉 OpenVINO Core 객체가 성공적으로 생성되었습니다.")
    print(f"OpenVINO 버전: {ov.__version__}")

    # 사용 가능한 디바이스 확인 (선택 사항)
    print("\n사용 가능한 추론 디바이스:")
    devices = ie.available_devices
    if devices:
        for device in devices:
            print(f"- {device}")
    else:
        print("- 감지된 디바이스가 없습니다.")

except ImportError:
    print("❌ 오류: 'openvino' 라이브러리를 찾을 수 없습니다. 설치가 필요합니다.")
    print("pip install openvino를 실행해 보세요.")
    sys.exit(1)
except Exception as e:
    print(f"❌ 오류: OpenVINO 초기화 중 예상치 못한 문제가 발생했습니다: {e}")
    sys.exit(1)

print("\nOpenVINO 설치 확인 스크립트 종료.")