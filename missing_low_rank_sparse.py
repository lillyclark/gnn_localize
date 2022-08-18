import numpy as np

def SigS_from_SigL(SigL, sigma):
    SigS = SigL**2
    SigS = -1*SigS
    SigS = SigS/(2*sigma**2)
    SigS = np.exp(SigS)
    SigS = SigS/(sigma**2)
    return SigS

def gradient_step(L, mu, sigma, c, z, N, s1, s2):
    U, SigL, Vt = np.linalg.svd(L)
    SigS = SigS_from_SigL(SigL, sigma)
    return L - mu*(U.dot(np.diag(SigL)).dot(np.diag(SigS)).dot(Vt) + c*L - (c*z+N).reshape(s1,s2))

def solve_e(y, M, c, A, z, lam):
    omega = y+M/c-A.dot(z)
    alpha = lam/c
    output = np.zeros_like(omega)
    output[omega<=-1*alpha] = omega[omega<=-1*alpha] + alpha
    output[omega>=alpha] = omega[omega>=alpha] - alpha
    return output

def solve_h(c, M, y, e, mask, L, N):
    return (M/c + y - e)*mask + L.flatten() - N/c

def solve_z(mask, h):
    return h*(mask==0) + h*(mask==1)*1/2

def update_M(M, c, y, mask, z, e):
    return M + c*(y-mask*z-e)

def update_N(N, c, z, L):
    return N + c*(z-L.flatten())

def MLSD(Y, mask, eps, sigma0, delta, mu, c, lam):
    s1, s2 = Y.shape
    y = Y.flatten()
    num_iters = 8
    z = np.zeros(s1*s2)
    N = np.zeros(s1*s2)
    M = np.zeros(s1*s2)
    L = Y
    k = 1
    sigma = sigma0

    round = 0
    while True:
        prev_L = L
        for i in range(num_iters):
            L = gradient_step(L, mu, sigma, c, z, N, s1, s2)
        sigma *= delta
        A = np.diag(mask)
        e = solve_e(y, M, c, A, z, lam)
        h = solve_h(c, M, y, e, mask, L, N)
        z = solve_z(mask, h)
        M = update_M(M, c, y, mask, z, e)
        N = update_N(N, c, z, L)

        if abs(np.sum(L - prev_L)) < eps:
            E = e.reshape(s1,s2)
            print("done in",round,"rounds")
            return L, E
        round += 1

    return L, E

if __name__=="__main__":
    locs = np.array([[0,0],[0,1],[1,0],[1,1],[1,2],[2,1],[2,2]])
    s1, s2 = locs.shape[0], locs.shape[0]
    Y = np.zeros((s1,s2))
    for i in range(s1):
        for j in range(s2):
            Y[i][j] = ((locs[j][0]-locs[i][0])**2+(locs[j][1]-locs[i][1])**2)**0.5
    Y = Y**2
    print("Y is low rank already")
    print(np.round(Y,2))
    print("rank of Y")
    print(np.linalg.matrix_rank(Y))
    mask = np.ones(s1*s2)
    print("mask is all ones (everything observed)")
    print(mask)

    eps = 1
    w, v = np.linalg.eig(Y)
    w = np.real(w)
    sigma0 = 4*np.max(w)
    delta = 0.9
    mu = 0.6
    c = 0.95
    lam = max(s1,s2)**0.5

    L,E = MLSD(Y, mask, eps, sigma0, delta, mu, c, lam)
    print("")
    print("Final solution")
    print("low rank L:")
    print(np.round(L,2))
    print("rank of matrix L:")
    print(np.linalg.matrix_rank(L))
    print("sparse E:")
    print(np.round(E,2))
