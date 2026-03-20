import hashlib
import random as rnd
import struct

import numpy as np
import torch
import torch.nn as nn

class NeuNAC:
    def __init__(self, klt_basis: np.ndarray, precision: int = 2):

        """
        Initialize NeuNAC instance.
        Args:
            klt_basis (np.ndarray): 32x32 KLT basis matrix (secret key).
            precision (int): Precision value 'p' used in bit extraction.
        """
        self.A = klt_basis
        self.p = precision

    def _flatten_parameters(self, model: nn.Module):
        flattened_weights = []
        for param in model.parameters():
            if param.requires_grad:
                flattened_weights.extend(param.detach().clone().flatten().tolist())
        flattened_weights = np.array(flattened_weights, dtype=np.float32)
        return flattened_weights

    def _split_into_pus(self, flattened_weights):
        """
        Splits a flattened weight list into Parameter Units (PUs),
        where each PU contains 16 float32 values.
        Args:
            flattened_weights (list): Flattened model weights (floats)
        Returns:
            list of PUs, each PU is a list of 16 float32s
        """
        usable_len = len(flattened_weights) - (len(flattened_weights) % 16)
        trimmed_weights = np.array(flattened_weights[:usable_len], dtype=np.float32)
        return [trimmed_weights[i:i + 16] for i in range(0, usable_len, 16)]

    def _generate_weu(self, pu):
        """
        Generate a 32-byte Watermark Embedding Unit (WEU) from a PU.
        Args:
            pu (list of float32): 16 float values
        Returns:
            list: 32-byte WEU = 16-byte MD5 fingerprint + 16-byte LSBs
        """
        msb_bytes, lsb_bytes = self._extract_msb_lsb_from_pu(pu)

        # Compute MD5 hash of MSBs
        fingerprint = hashlib.md5(msb_bytes).digest()  # returns 16-byte bytes object

        # Combine fingerprint and LSBs
        weu = list(fingerprint + lsb_bytes) 
        assert len(weu) == 32

        return weu

    def _extract_msb_lsb_from_pu(self, pu):
        """
        Given a Parameter Unit (PU) of 16 float32 values,
        extract 3 MSBs and 1 LSB per float.
        Returns:
            msb_bytes: bytearray of 48 bytes (3 MSBs x 16)
            lsb_bytes: bytearray of 16 bytes (1 LSB x 16)
        """
        assert len(pu) == 16, "PU must contain exactly 16 float32 values"

        msb_bytes = bytearray()
        lsb_bytes = bytearray()

        for f in pu:
            float_bytes = struct.pack('>f', f)  # big-endian float32 = 4 bytes
            msb_bytes.extend(float_bytes[:3])  # first 3 bytes
            lsb_bytes.append(float_bytes[3])  # last byte
        return msb_bytes, lsb_bytes

    def _generate_global_watermark(self, pus, w_bits=None):
        """
        Generate the global watermark bitstring w from the MSBs of all PUs.
        Args:
            pus (list): List of PUs (each PU = 16 float32 values)
            w_bits (int): Optional. If specified, truncate the bitstring to this many bits
        Returns:
            str: Bitstring of the global watermark
        """
        total_msb_bytes = bytearray()

        for pu in pus:
            msb_bytes, _ = self._extract_msb_lsb_from_pu(pu)
            total_msb_bytes.extend(msb_bytes)

        w_bits = len(pus)
        digest = hashlib.shake_256(total_msb_bytes).digest((w_bits + 7) // 8)

        # Convert digest to bitstring
        bitstring = ''.join(f'{byte:08b}' for byte in digest)

        if w_bits is not None:
            bitstring = bitstring[:w_bits]

        return bitstring

    def _embed_bit_in_weu(self, weus, global_watermark, A, mu, p):
        new_weus, extracted = [], []
        j = 0
        for i, weu in enumerate(weus):
            print("j:", j)
            j += 1
            c0 = self._compute_klt_coefficients(weu, A, mu)[0]
            bit = self._extract_bit_from_coefficient(c0, p)
            if bit != int(global_watermark[i]):
                weu = self._store_bits_with_ga(
                    weu=weu,
                    target_bit=int(global_watermark[i]),
                    A=A,
                    mu=mu,
                    p=p,
                )
                bit = int(global_watermark[i])

            new_weus.append(weu)
            extracted.append(bit)

        return new_weus, extracted

    def _store_bits_with_ga(self, weu, target_bit, A, mu, p,
                            pop_size: int = 100,
                            max_gen: int = 2000,
                            pc: float = 0.8,
                            pm: float = 0.05,
                            penalty: int = 10,
                            ):
        """
        Return a *new* WEU whose first KLT-coeff bit (Eq. 4) equals 'target_bit',
        while keeping the fingerprint half unchanged and the distortion minimal.
        Only the last 16 bytes (LSBs) are allowed to change.
        """
        # --- fixed / mutable split -------------------------------------------
        base = np.asarray(weu, dtype=np.int16)  # allow negative deltas
        frozen = base[:16]  # 16-byte MD5 fingerprint
        mutable = base[16:]  # 16 bytes we may alter

        # --- helpers for GA ----------------------------------------------------------
        def apply(delta: np.ndarray) -> np.ndarray:
            """Add chromosome 'delta' to mutable bytes, clip to 0...255."""
            return np.concatenate((frozen, np.clip(mutable + delta, 0, 255)))

        def fitness(delta: np.ndarray) -> int:
            """Lower is better: |delta| + penalty if bit is wrong."""
            cand = apply(delta)
            c0 = self._compute_klt_coefficients(cand.tolist(), A, mu)[0]
            bit_ok = self._extract_bit_from_coefficient(c0, p) == target_bit
            err = 0 if bit_ok else penalty
            return int(np.sum(np.abs(delta)) + err)

        # --- initial population ----------------------------------------------
        pop = [np.zeros(16, dtype=np.int16)]
        for _ in range(pop_size - 1):
            d = np.zeros(16, dtype=np.int16)
            flip = rnd.randrange(16)
            d[flip] = rnd.choice([-1, 1])
            pop.append(d)

        best = min(pop, key=fitness)

        # --- evolutionary loop -----------------------------------------------
        for _ in range(max_gen):
            # roulette-wheel selection on *inverse* fitness
            fit_vals = np.array([fitness(ind) for ind in pop], dtype=np.float32)
            probs = (fit_vals.max() + 1) - fit_vals
            probs /= probs.sum()

            # offspring generation
            children = []
            while len(children) < int(pop_size * 0.95):
                p1, p2 = rnd.choices(pop, k=2, weights=probs, cum_weights=None)
                child = p1.copy()

                # crossover
                if rnd.random() < pc:
                    cx = rnd.randrange(1, 16)
                    child[:cx] = p2[:cx]

                # mutation
                if rnd.random() < pm:
                    for idx in rnd.sample(range(16), k=rnd.randrange(1, 4)):
                        child[idx] += rnd.choice([-4, -3, -2, -1, 1, 2, 3, 4])

                children.append(child)

            pop = sorted(pop + children, key=fitness)[:pop_size]
            if fitness(pop[0]) < fitness(best):
                best = pop[0]

            # stop as soon as the bit is correct
            if fitness(best) < penalty:
                break

        cand_bit = self._extract_bit_from_coefficient(
            self._compute_klt_coefficients(apply(best), A, mu)[0], p
        )
        if cand_bit != target_bit:
            print(
                f"[GA warning ---------------------------- ] Could not embed bit {target_bit} after {max_gen} generations.")

        else:
            print('GA found the bits to change.')
        return apply(best).astype(np.uint8).tolist()

    def _compute_klt_coefficients(self, weu, A, mu):
        """
        Compute KLT coefficients for each WEU in the list.

        Args:
            weus (list): List of WEUs (each a list of 32 bytes)
            A (np.ndarray): 32x32 KLT basis matrix (float32)
            mu (np.ndarray): 32D mean vector (float32)

        Returns:
            list of np.ndarray: Each element is a 32D vector of KLT coefficients
        """
        assert len(weu) == 32, "Each WEU must be 32 bytes"
        v = np.array(weu, dtype=np.float32)
        centered = v - mu
        c = A @ centered
        return c

    def _extract_bit_from_coefficient(self, c_i: float, p: int) -> int:
        """
        Extract a watermark bit from a KLT coefficient using NeuNAC Formula (4).
        Args:
            c_i (float): Selected KLT coefficient
            p (int): Precision

        Returns:
            int: Extracted bit (0 or 1)
        """
        return int(round(c_i * (2 ** -p)) % 2)

    def _apply_weus_to_weights(self, original_flat, new_weus: list):
        """
        Create a *new* flattened weight array whose LSBs come from `new_weus`.
        The first len(new_weus)·16 floats are updated; any tail floats (if length
        not divisible by 16) are left untouched.
        """
        updated = original_flat.copy()

        for pu_idx, weu in enumerate(new_weus):
            base = pu_idx * 16
            lsb_block = weu[16:]  # 16 mutated bytes
            for i, new_byte in enumerate(lsb_block):
                f_idx = base + i  # index of the float in flat array
                updated[f_idx] = self._replace_lsb(updated[f_idx], int(new_byte))

        return updated

    def _replace_lsb(self, f: float, new_byte: int) -> float:
        """Return `f` but with its least-significant byte replaced by new_byte."""
        b = bytearray(struct.pack('>f', f))  # big-endian to match extraction
        b[3] = new_byte  # overwrite LSB
        return struct.unpack('>f', b)[0]

    def _reload_model_from_flat(self, model: nn.Module, flat):
        """
        Overwrite `model`s parameters with floats from `flat` (torch expects row-major).
        """
        cursor = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                slice_ = flat[cursor:cursor + numel]
                cursor += numel
                param.data.copy_(torch.from_numpy(slice_.reshape(param.shape)))

    # --- HELPER MODULES END --- #

    def embed_watermark(self, model: nn.Module):
        """
        Embed a NeuNAC watermark into `model`.

        Returns
        -------
        watermarked_model : the same instance, modified in-place
        klt_mean          : it is used to extract the watermark.
        """
        flattened_weights = self._flatten_parameters(model)
        pus = self._split_into_pus(flattened_weights)
        print(len(pus))

        weus = [self._generate_weu(pu) for pu in pus]
        mu = np.mean(np.array(weus, dtype=np.float32), axis=0)  # This will be returned so that the user can store.
        global_watermark = self._generate_global_watermark(pus=pus)
        new_weus, _ = self._embed_bit_in_weu(weus, global_watermark, self.A, mu, self.p)
        new_flat = self._apply_weus_to_weights(flattened_weights, new_weus)
        self._reload_model_from_flat(model, new_flat)

        return model, mu

    def extract_watermark(self, model: nn.Module, klt_matrix_key, klt_mean) -> str:
        """
        Extract watermark bits from a model's weights.

        Args:
            model_weights (nn.Module) : The torch model
            klt_matrix_key: Necessary key_1 used in embedding watermark, produced by user.
            klt_mean: Necessary key_2, produced by embedding watermark.

        Returns:
            Bitstring: Extracted watermark from the model.
        """
        flattened_weights = self._flatten_parameters(model)
        pus = self._split_into_pus(flattened_weights)
        weus = [self._generate_weu(pu) for pu in pus]
        bits = [
            self._extract_bit_from_coefficient(self._compute_klt_coefficients(w, klt_matrix_key, klt_mean)[0], self.p)
            for w in weus
        ]
        return pus, bits

    def verify_watermark(self, pus, extracted_bits) -> bool:
        """
        Compare extracted watermark with expected watermark.

        Args:
            model_weights (list): Flattened list of model weights (floats).
            expected_watermark (str): Bitstring of expected watermark.

        Returns:
            bool: True if watermark matches, False otherwise.
        """
        expected = self._generate_global_watermark(pus=pus)
        expected_bits = [int(b) for b in expected[:len(extracted_bits)]]
        ok = extracted_bits == expected_bits

        # Save to text file
        save_path = "watermark_verification.txt"
        with open(save_path, "w") as f:
            f.write(f"Match: {ok}\n")

        return ok

