
import numpy as np

np.random.seed(20261123)

# 실험 설정
blocks = 20000
bits = np.random.randint(0, 2, (blocks, 4))

snr_values = [0, 2, 4, 6, 8]

# 1. 해밍 (7,4) 부호화
encoded = np.zeros((blocks, 7), dtype=int)

# 데이터 비트 배치
encoded[:, 2] = bits[:, 0]  # D1
encoded[:, 4] = bits[:, 1]  # D2
encoded[:, 5] = bits[:, 2]  # D3
encoded[:, 6] = bits[:, 3]  # D4

# 짝수 패리티 계산
encoded[:, 0] = (
    encoded[:, 2] ^ encoded[:, 4] ^ encoded[:, 6]
)
encoded[:, 1] = (
    encoded[:, 2] ^ encoded[:, 5] ^ encoded[:, 6]
)
encoded[:, 3] = (
    encoded[:, 4] ^ encoded[:, 5] ^ encoded[:, 6]
)

# 2. 잡음 조건별 실험
for db in snr_values:

    ebn0 = 10 ** (db / 10)
    sigma = np.sqrt(1 / (2 * ebn0))

    # BPSK 변조
    signal = 1 - 2 * encoded

    # AWGN 잡음 추가
    noise = np.random.normal(
        0, sigma, signal.shape
    )
    received = signal + noise

    # 3. 복조
    decoded = (received < 0).astype(int)

    # 4. 신드롬 계산
    s1 = (
        decoded[:, 0] ^ decoded[:, 2]
        ^ decoded[:, 4] ^ decoded[:, 6]
    )

    s2 = (
        decoded[:, 1] ^ decoded[:, 2]
        ^ decoded[:, 5] ^ decoded[:, 6]
    )

    s4 = (
        decoded[:, 3] ^ decoded[:, 4]
        ^ decoded[:, 5] ^ decoded[:, 6]
    )

    syndrome = s1 + 2*s2 + 4*s4

    # 5. 오류 위치 수정
    corrected = decoded.copy()

    rows = np.where(syndrome != 0)[0]

    corrected[
        rows, syndrome[rows] - 1
    ] ^= 1

    # 6. 원본 데이터 추출
    recovered = corrected[:, [2, 4, 5, 6]]

    # 7. 오류율 계산
    errors = np.sum(recovered != bits)
    ber = errors / bits.size

    # 블록 복구 성공률
    success = np.all(
        recovered == bits, axis=1
    ).sum() / blocks

    print(
        f"{db} dB | 오류: {errors} | "
        f"BER: {ber:.6f} | "
        f"블록 성공률: {success*100:.3f}%"
    )
