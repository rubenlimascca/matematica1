import numpy as np

A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
print(A)

print("-------------")

b = np.array([8, -11, -3])
print(b)

print("-------------")

N = len(b)
print(N)

print("-------------")

x = np.zeros(N)
print(x)

print("-------------")

for i in range(N - 1):
    b[i] = b[i] / A[i][i]
    A[i] = A[i] / A[i][i]
    for k in range(i + 1, N):
        b[k] = b[k] - A[k][i] * b[i]
        A[k] = A[k] - A[k][i] * A[i]

i = N - 1
b[i] = b[i] / A[i][i]
A[i] = A[i] / A[i][i]

# e = np.linalg.norm(A @ x - b)
print(A, b)
