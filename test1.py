
import numpy as np

np.random.seed(20261123)

# 실험 설정
blocks = 20000
bits = np.random.randint(0, 2, (blocks, 4))

snr_values = [0, 2, 4, 6, 8]

for db in snr_values:
    # dB를 실제 비율로 변환
    ebn0 = 10 ** (db / 10)

    # 잡음의 표준편차
    sigma = np.sqrt(1 / (2 * ebn0))

    # BPSK 변조
    signal = 1 - 2 * bits

    # AWGN 잡음 추가
    noise = np.random.normal(
        0, sigma, signal.shape
    )
    received = signal + noise

    # 복조
    decoded = (received < 0).astype(int)

    # 오류 계산
    errors = np.sum(decoded != bits)
    ber = errors / bits.size

    # 블록 전체 복구 성공률
    success = np.all(
        decoded == bits, axis=1
    ).sum() / blocks

    print(
        f"{db} dB | 오류: {errors} | "
        f"BER: {ber:.6f} | "
        f"블록 성공률: {success*100:.3f}%"
    )
