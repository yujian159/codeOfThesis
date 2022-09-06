import epoch as epoch
import numpy as np
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp

from tensorly.tenalg import khatri_rao
from tqdm import tqdm


def cp_bp(tensor: np.ndarray, R=1, lr=1e-2, max_iter=100):
    N = tl.ndim(tensor)
    # Step 1
    lbd, A = initialize_cp(tensor, R, init='random', svd='numpy_svd',
                           random_state=0,
                           normalize_factors=True)

    for epoch in range(max_iter):
        # Step 2
        tensor_pred = tl.fold(np.matmul(np.matmul(A[0], np.diag(lbd)),
                                        khatri_rao(A, skip_matrix=0).T),
                              mode=0,
                              shape=tensor.shape)
        theta = (tensor - tensor_pred)
        grad_A = []

        # Step 3
        for n in range(N):
            grad_A.append(np.zeros_like(A[n]))
            grad_A[n] = np.matmul(tl.unfold(theta, n), khatri_rao(A, skip_matrix=n))

        # Step 4
        for n in range(N):
            A[n] = A[n] + lr * grad_A[n]

        # Step 5
        loss = np.sum(0.5 * np.square(tl.unfold(theta, 0)))
        print("epoch {}: loss={}".format(epoch, loss))

    return A, lbd


if __name__ == '__main__':
    np.random.seed(10086)
    inpt = tl.tensor(np.random.random((3, 3, 3)), dtype=np.float32)
    A, lbd = cp_bp(inpt, R=5, lr=1e-2, max_iter=100)
    tensor_pred_cp = tl.fold(np.matmul(np.matmul(A[0], np.diag(lbd)),
                                       khatri_rao(A, skip_matrix=0).T),
                             mode=0,
                             shape=inpt.shape)

    print("tensor_pred_cp: ", tl.norm(inpt - tensor_pred_cp), epoch)
