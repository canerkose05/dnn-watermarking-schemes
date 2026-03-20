import hashlib
import numpy as np
import scipy.sparse as sp
from pyldpc import make_ldpc, decode, get_message
from typing import Sequence, Tuple, Dict


class TattooedWatermarker:
    """TATTOOED spread-spectrum watermark with an *outer* LDPC layer.
    """

    # ---------- class‑wide cache to avoid rebuilding the same LDPC matrices ----
    _LDPC_CACHE: Dict[Tuple[int, int, float, bytes], Tuple[np.ndarray, np.ndarray]] = {}

    def __init__(
            self,
            key: bytes,
            *,
            ratio: float = 0.03,
            gamma: float = 9e-4,
            ldpc_rate: float = 0.5,
    ) -> None:
        """Args
        -----
        key
            Secret byte string. Seeds and LDPC matrices are derived from it.
        ratio
            Fraction (0, 1] of model parameters to watermark
        gamma
            Chip amplitude (9x10^-4 in the paper).
        ldpc_rate
            *k/n* of the (d_v, d_c) LDPC
        """
        if not (0.0 < ratio <= 1.0):
            raise ValueError("ratio must be in (0,1]")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if not (0.0 < ldpc_rate < 1.0):
            raise ValueError("ldpc_rate must be in (0,1)")

        self.secret = key  # store for LDPC helpers
        self.ratio = float(ratio)
        self.gamma = np.float32(gamma)
        self.ldpc_rate = float(ldpc_rate)

        # derive two 256‑bit sub‑keys
        self.params_seed, self.code_seed = self._seed_gen(key)

    def embed_watermark(self, weights: np.ndarray, bits: Sequence[int]) -> np.ndarray:
        """Return a **copy** of *weights* with the watermark embedded.

        The caller supplies *plain* bits (payload length *k*).  Internally we
        LDPC-encode them -> codeword length *n*, then apply the CDMA spreader
        as in the paper
        """
        msg_bits = np.asarray(bits, dtype=np.uint8)
        if not np.isin(msg_bits, [0, 1]).all():
            raise ValueError("Bits must be 0 or 1")

        H, G = self._get_ldpc(msg_bits.size)
        code_bits = self._ldpc_encode(msg_bits, G)  # length n

        w_marked = weights.copy()
        m = self._to_bipolar(code_bits)  # +1 or -1 chips per bit
        R, idx = self._select_indices(len(w_marked))

        rng_code = np.random.default_rng(self.code_seed)
        code_buffer = rng_code.choice([-1, 1], size=(m.size, R)).astype(np.float32)

        # vectorised update for speed
        w_marked[idx] += self.gamma * (m.astype(np.float32) @ code_buffer)
        return w_marked

    def extract_watermark(self, weights_wtm: np.ndarray, num_bits: int) -> np.ndarray:
        """Recover *num_bits* payload bits from a watermarked weight vector."""
        H, G = self._get_ldpc(num_bits)
        # CDMA hard‑decision detection ------------------------------------
        R, idx = self._select_indices(len(weights_wtm))
        rng_code = np.random.default_rng(self.code_seed)
        code_buffer = rng_code.choice([-1, 1], size=(self._ldpc_n(num_bits), R)).astype(np.float32)
        y = code_buffer @ weights_wtm[idx].astype(np.float32)
        coded_bits = (y <= 0).astype(np.uint8)  # sign test -> 0/1 length n

        # LDPC decode back to the original k‑bit payload ------------------
        msg_hat = self._ldpc_decode(coded_bits, H, G, k_orig=num_bits)

        return msg_hat

    def verify_watermark(
            self,
            original_bits: Sequence[int],
            extracted_bits: Sequence[int],
            *,
            threshold: float = 0.0,
    ) -> Tuple[bool, float]:
        """Return (is_valid, BER).  Passes if BER <= *threshold*."""
        a = np.asarray(original_bits, dtype=np.uint8)
        b = np.asarray(extracted_bits, dtype=np.uint8)
        if a.shape != b.shape:
            raise ValueError("Bit vectors must have same length")
        ber = float(np.mean(a != b))
        return ber <= threshold, ber

    # ------------------------------------------------------------------
    # private helpers (seed, LDPC, etc.)
    # ------------------------------------------------------------------
    @staticmethod
    def _seed_gen(key: bytes) -> Tuple[int, int]:
        digest = hashlib.sha512(key).digest()
        params_seed = int.from_bytes(digest[:32], "big")
        code_seed = int.from_bytes(digest[32:], "big")
        return params_seed, code_seed

    @staticmethod
    def _to_bipolar(bits: np.ndarray) -> np.ndarray:
        return 1 - 2 * bits  # 0-> +1, 1-> −1

    def _select_indices(self, total_len: int) -> Tuple[int, np.ndarray]:
        R = int(total_len * self.ratio)
        if R == 0:
            raise ValueError("ratio too small for this model (R rounded to 0)")
        rng = np.random.default_rng(self.params_seed)
        idx = rng.choice(total_len, size=R, replace=False)
        return R, idx

    # ---------- LDPC layer -------------------------------------------------
    @staticmethod
    def _rng_from_secret(secret: bytes, tag: str) -> np.random.Generator:
        digest = hashlib.sha512(secret + tag.encode()).digest()
        seed = int.from_bytes(digest[:8], "big")
        return np.random.default_rng(seed)

    def _ldpc_n(self, k_msg: int) -> int:
        return int(np.ceil(k_msg / self.ldpc_rate))

    def _get_ldpc(self, k_msg: int):
        key = (k_msg, int(self.ldpc_rate * 1e6), self.secret)

        if key not in self._LDPC_CACHE:
            rng = self._rng_from_secret(self.secret, "ldpc")

            # --- choose n so that d_c divides it
            d_v = 3
            d_c = int(d_v / self.ldpc_rate)  # 6 when rate = 0.5
            n_code = int(np.ceil(k_msg / self.ldpc_rate))
            if n_code % d_c:  # round up to next multiple
                n_code += d_c - n_code % d_c  # 512 -> 516

            # --- build the matrices 
            try:
                H, G = make_ldpc(n_code, d_v, d_c,
                                 systematic=True, sparse=True,
                                 seed=int(rng.integers(2 ** 32)))
            except TypeError:  # pyldpc 0.4.x fallback
                H, G = make_ldpc(n_code, d_v, d_c,
                                 systematic=True, sparse=True,
                                 random_state=rng)

            if not sp.issparse(H):
                H = sp.csr_matrix(H)
            if not sp.issparse(G):
                G = sp.csr_matrix(G)

            self._LDPC_CACHE[key] = (H, G)

        return self._LDPC_CACHE[key]

    @staticmethod
    def _ldpc_encode(msg_bits: np.ndarray, G) -> np.ndarray:
        """Return a length-n codeword (0/1), padding msg_bits if G expects more rows."""
        k_full = G.shape[0]
        if msg_bits.size > k_full:
            raise ValueError("payload longer than LDPC message length")
        # pad with zeros when k' > k   (pyldpc ignores parity rows anyway)
        padded = np.zeros(k_full, dtype=np.uint8)
        padded[: msg_bits.size] = msg_bits
        return (padded @ G.toarray()) % 2

    @staticmethod
    def _ldpc_decode(code_bits, H, G, k_orig: int, snr_db: int = 8):
        """
        Hard-decision code_bits -> LDPC decode -> original k bits.
        H, G may be dense or sparse.
        """
        # 1) cast to float LLRs and reshape as column vector (n,1)
        llr = (1.0 - 2.0 * code_bits).astype(np.float64).reshape(-1, 1)

        # 2) make sure H is a *dense* ndarray; decode() stumbles on sparse
        H_dense = H.toarray() if hasattr(H, "toarray") else np.asarray(H)

        cw_hat = decode(H_dense, llr, snr=50, maxiter=200)
        cw_hat = np.asarray(cw_hat).ravel() % 2  # flat 1‑D 0/1 vector

        full_msg = get_message(G.toarray()
                               if hasattr(G, "toarray") else np.asarray(G),
                               cw_hat).astype(np.uint8)

        return full_msg[:k_orig]  # drop pad bits (e.g. 258 -> 256)
