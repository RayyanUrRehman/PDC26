"""Independent i-k-j reference for the seeded generator. Prints the strided checksum."""

N = 32
state = 42


def fill(count: int) -> list[float]:
    global state
    out = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        unit = (state >> 11) * (1.0 / 9007199254740992.0)
        out.append(unit * 2.0 - 1.0)
    return out


A = fill(N * N)
B = fill(N * N)
C = [0.0] * (N * N)
for i in range(N):
    for k in range(N):
        aik = A[i * N + k]
        for j in range(N):
            C[i * N + j] += aik * B[k * N + j]
checksum = sum(v for idx, v in enumerate(C) if idx % 1024 == 0)
print(f"{checksum:.6e}")
