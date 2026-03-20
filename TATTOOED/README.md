## TATTOOED: A Robust Deep Neural Network Watermarking Scheme based on Spread-Spectrum Channel Coding

### Scheme
Secret key → two seeds → LDPC encode payload → CDMA/spread spectrum embed into a subset of weights → extract by correlation → LDPC decode back to original bits

### Input
- Weights - 1D vector of model parameters W
- Payload Bits - Payload bits of length k (e.g., 256)
- Key - Secret key s
- Ratio - How many weights are used for watermarking R
- Gamma - Watermark strength γ
- Ldpc_rate - Outer code rate

### Embedding
1. Derive two seeds from the key s.
    - Seed 1 selects the weight indices idx
    - Seed 2 generates spreading codes
2. LDPC encodes the payload bits (length k) into a codeword of length n, n > k.
3. Convert codeword bits to bipolar symbols m ∈ {-1, +1} via 0-> +1, 1 -> -1
4. Generate a spreading code vector c<sub>i</sub> ∈ {-1, +1}<sup>R</sup> for each encoded m<sub>i</sub> bit using seed 2, forming C ∈ {-1, +1}<sup>nxR</sup>
5. Add the watermark signal into the weights selected by seed 1:
    - W<sub>wtm</sub>[idx] =W[idx] + γm<sup>T</sup>C 

### Extraction
1. Derive two seeds from the key s.
    - Seed 1 selects the weight indices idx
    - Seed 2 generates spreading codes
2. Regenerate the same spreading code matrix C ∈ {-1, +1}<sup>nxR</sup> where each row c<sub>i</sub> is the spreading code corresponding to one LDPC-encoded bit.
3. Restrict the watermarked weights to the selected indices: W<sub>wtm</sub>[idx]
4. For each encoded bit i=1,...,n compute the correlation
    - y<sub>i</sub> = c<sub>i</sub><sup>T</sup>W<sub>wtm</sub>[idx]
5. Recover a noisy estimate of the bipolar symbols via hard decision:
    - m<sub>i</sub>^ = +1, if y<sub>i</sub> > 0
    - m<sub>i</sub>^ = -1, if y<sub>i</sub> <= 0
6. Map the bipolar symbols back to binary form:
    - b<sub>i</sub>^ = 0, if m<sub>i</sub>^ = +1
    - b<sub>i</sub>^ = 1, if m<sub>i</sub>^ = -1
7. Apply LDPC decoding to the recovered codeword bits b^ to reconstruct the original payload bits of length k.
