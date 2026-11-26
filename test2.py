
import numpy as np

# 실험 설정
np.random.seed(20261123)

blocks = 20000
bits = np.random.randint(0, 2, (blocks, 4))

snr_values = [0, 2, 4, 6, 8]
repeat_values = [3, 5]

for db in snr_values:
    ebn0 = 10 ** (db / 10)
    sigma = np.sqrt(1 / (2 * ebn0))

    print(f"\n===== {db} dB =====")

    for r in repeat_values:

        # 각 비트를 r회 반복
        encoded = np.repeat(
            bits[:, :, np.newaxis],
            r,
            axis=2
        )

        # BPSK 변조
        signal = 1 - 2 * encoded

        # AWGN 잡음 추가
        noise = np.random.normal(
            0, sigma, signal.shape
        )

        received = signal + noise

        # 복조
        decoded = (received < 0).astype(int)

        # 다수결 복호화
        recovered = (
            np.sum(decoded, axis=2) > r // 2
        ).astype(int)

        # 오류 계산
        errors = np.sum(recovered != bits)
        ber = errors / bits.size

        # 블록 성공률
        success = np.all(
            recovered == bits, axis=1
        ).sum() / blocks

        print(
            f"{r}회 반복 | "
            f"오류: {errors} | "
            f"BER: {ber:.6f} | "
            f"블록 성공률: {success*100:.3f}%"
        )
