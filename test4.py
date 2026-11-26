
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 실험 기본 설정
# ==========================================

BLOCKS = 20000
K = 4
SNR_DB_LIST = [0, 2, 4, 6, 8]
SEED = 20261123

rng = np.random.default_rng(SEED)

# 전송 효율 R = 정보 비트 수 / 전송 비트 수
RATES = {
    "기본 통신": 1.0,
    "3회 반복": 1 / 3,
    "5회 반복": 1 / 5,
    "해밍(7,4)": 4 / 7
}

# 동일한 정보 데이터 사용
data = rng.integers(0, 2, size=(BLOCKS, K))


# ==========================================
# 2. 부호화 함수
# ==========================================

def encode_basic(bits):
    return bits.copy()


def encode_repeat(bits, n):
    return np.repeat(bits, n, axis=1)


def encode_hamming(bits):
    d1, d2, d3, d4 = bits.T

    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4

    code = np.column_stack([
        p1, p2, d1, p4, d2, d3, d4
    ])

    return code


# ==========================================
# 3. BPSK 변조 및 AWGN 채널
# ==========================================

def transmit(bits, snr_db, rate):
    symbols = (1 - 2 * bits) * np.sqrt(rate)

    gamma = 10 ** (snr_db / 10)

    sigma = np.sqrt(1 / (2 * gamma))

    noise = rng.normal(
        0,
        sigma,
        size=symbols.shape
    )

    received = symbols + noise

    return received


# ==========================================
# 4. 복호화 함수
# ==========================================

def decode_basic(received):
    return (received < 0).astype(int)


def decode_repeat(received, n):
    # 반복된 비트들을 원래 비트 단위로 묶음
    grouped = received.reshape(
        received.shape[0],
        K,
        n
    )

    # 다수결 복호화
    decoded = (np.sum(grouped, axis=2) < 0).astype(int)

    return decoded


def decode_hamming(received):
    # BPSK 경판정
    code = (received < 0).astype(int)

    # 신드롬 계산
    s1 = code[:, 0] ^ code[:, 2] ^ code[:, 4] ^ code[:, 6]
    s2 = code[:, 1] ^ code[:, 2] ^ code[:, 5] ^ code[:, 6]
    s4 = code[:, 3] ^ code[:, 4] ^ code[:, 5] ^ code[:, 6]

    syndrome = s1 + 2 * s2 + 4 * s4

    # 단일 비트 오류 수정
    for position in range(1, 8):
        mask = syndrome == position
        code[mask, position - 1] ^= 1

    # 원래 정보 비트 추출
    decoded = code[:, [2, 4, 5, 6]]

    return decoded


# ==========================================
# 5. 실험 수행
# ==========================================

results = []

for snr_db in SNR_DB_LIST:

    print(f"\n========== {snr_db} dB ==========")

    # 각 방식의 부호화
    encoded_data = {
        "기본 통신": encode_basic(data),
        "3회 반복": encode_repeat(data, 3),
        "5회 반복": encode_repeat(data, 5),
        "해밍(7,4)": encode_hamming(data)
    }

    for method, encoded in encoded_data.items():

        rate = RATES[method]

        # 잡음 채널 통과
        received = transmit(
            encoded,
            snr_db,
            rate
        )

        # 복호화
        if method == "기본 통신":
            decoded = decode_basic(received)

        elif method == "3회 반복":
            decoded = decode_repeat(received, 3)

        elif method == "5회 반복":
            decoded = decode_repeat(received, 5)

        elif method == "해밍(7,4)":
            decoded = decode_hamming(received)

        # 오류 계산
        errors = np.sum(data != decoded)
        total_bits = data.size

        ber = errors / total_bits

        # 블록 전체가 정확하게 복구된 비율
        block_success = np.mean(
            np.all(data == decoded, axis=1)
        ) * 100

        efficiency = rate * 100

        results.append({
            "SNR(dB)": snr_db,
            "통신 방식": method,
            "오류 비트 수": errors,
            "BER": ber,
            "블록 성공률(%)": block_success,
            "전송 효율(%)": efficiency
        })

        print(
            f"{method:10s} | "
            f"오류: {errors:5d} | "
            f"BER: {ber:.6f} | "
            f"블록 성공률: {block_success:.3f}%"
        )


# ==========================================
# 6. 결과 표 출력
# ==========================================

df = pd.DataFrame(results)

print("\n\n========== 전체 실험 결과 ==========")

print(
    df.to_string(
        index=False,
        formatters={
            "BER": "{:.6f}".format,
            "블록 성공률(%)": "{:.3f}".format,
            "전송 효율(%)": "{:.1f}".format
        }
    )
)

# CSV 파일 저장
df.to_csv(
    "experiment4_results.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 7. BER 비교 그래프
# ==========================================

plt.figure(figsize=(9, 6))

for method in RATES.keys():

    subset = df[df["통신 방식"] == method]

    plt.plot(
        subset["SNR(dB)"],
        subset["BER"],
        marker="o",
        label=method
    )

plt.yscale("log")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER Comparison of Error Correction Methods")
plt.xticks(SNR_DB_LIST)
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================
# 8. 블록 성공률 비교 그래프
# ==========================================

plt.figure(figsize=(9, 6))

for method in RATES.keys():

    subset = df[df["통신 방식"] == method]

    plt.plot(
        subset["SNR(dB)"],
        subset["블록 성공률(%)"],
        marker="o",
        label=method
    )

plt.xlabel("SNR (dB)")
plt.ylabel("Block Success Rate (%)")
plt.title("Block Recovery Success Rate")
plt.xticks(SNR_DB_LIST)
plt.ylim(0, 105)
plt.grid(True, linestyle="--")
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================
# 9. 전송 효율 비교
# ==========================================

print("\n========== 전송 효율 비교 ==========")

for method, rate in RATES.items():
    print(f"{method}: {rate * 100:.2f}%")
