## NeuNAC: A novel fragile watermarking algorithm for integrity protection of neural networks

### Scheme
Secret key(s) → split model weights into small groups → build editable bytes → generate watermark bits from stable parts of the model → adjust least important bytes → put bytes back into weights → read bits using the same secret → recompute expected bits from the model

### Input
- A trained neural network model
- Secret key A: KLT basis matrix – size 32x32,
- Secret key p: a precision parameter
- Stored value μ: Returned by embedding procedure and required for extraction

### Embedding

1. Flatten trainable model weights
2. Split the flattened weights into 16 floating-point value chunks. Each value is 4 bytes and 16 value*4 bytes = 64 bytes in total (Each chunk is called “parameter units” - PU)
3. For each PU:
    - Extract the first three bytes of each float value (most significant bytes - MSB)
    - Extract the last byte of each float value (least significant byte - LSB)
    - Compute an MD5 hash over all extracted MSBs (input 16 value * 3 bytes= 48 bytes, output a 16-byte hash)
    - Combine the hash and the least significant bytes (16 value * 1 bytes = 16 bytes) to form a 32-byte watermark embedding unit WEU.
4. Generate a global watermark bit sequence by hashing the most significant bytes of all PUs (the same first three bytes of each float for all PUs)
5. Compute average of all WEUs (bytewise) and store it as mu
6. For each WEU that is calculated in step 3:
    - Compute KLT coefficients c = A @ (WEU - μ) 
    - Extract bit from the first coefficient c0 using p (secret key 2)
    - If c0 != global_watermark[WEU_Index] → run Genetic Algorithm  to modify only the last 16 bytes until the bit matches.
7. Write modified WEUs back into the weights
8. Embedding returns μ which must be stored and provided during extraction

### Extraction
  Flatten → Split into PUs → Create WEUs (same as embedding)
  Extract one bit per WEU using keys A, p and mu by applying the KLT transform.

### Verification
  Recompute expected watermark bits from the model MSBs and compare to extracted bits (integrity check, not ownership claim)
