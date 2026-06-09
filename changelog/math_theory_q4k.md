# Math Theory — Q2_K, Q3_K, Q4_K, Q5_K, Q6_K Dequantisation

This document mirrors the style of the existing Q8_0/Q4_0 theory and covers
the three K-quant formats added in the GGQR-CF-mmap dequantisation engine.

---

## Background: K-Quants vs. Legacy Quants

Legacy formats (Q4_0, Q8_0) use a single f16 scale per 32-element block with
symmetric quantisation (zero_point = 0). K-quants introduce a **superblock**
hierarchy: one superblock covers 256 elements and carries a global scale pair
`(d, dmin)`, while each sub-block inside the superblock carries its own 6-bit
scale and min. This two-level structure allows the quantiser to adapt more
tightly to the local distribution of weights, reducing reconstruction error
without increasing the average bits-per-weight.

---

## Q4_K

### Superblock structure

| Region  | Size      | Description                                      |
|---------|-----------|--------------------------------------------------|
| `d`     | 2 bytes   | f16 — superblock scale factor                    |
| `dmin`  | 2 bytes   | f16 — superblock min factor                      |
| `scales`| 12 bytes  | 8 × 6-bit sub-block scales + 8 × 6-bit sub-block mins, packed |
| nibbles | 128 bytes | 256 × 4-bit unsigned values, two per byte        |

**Total: 144 bytes per 256 elements → 4.5 bits/weight**

### Sub-block scale/min packing (GGML `get_scale_min_k4`)

The 12-byte `scales` region encodes 8 pairs of 6-bit values using the
following scheme (indices into the `scales` byte array):

```
For j in 0..4:
  scale[j] = scales[j]     & 0x3F
  min[j]   = scales[j + 4] & 0x3F

For j in 4..8:
  scale[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
  min[j]   = (scales[j + 4] >> 4)   | ((scales[j    ] >> 6) << 4)
```

The high 2 bits of `scales[0..7]` carry the overflow bits for `scale[4..7]`
and `min[4..7]`. This packs 16 × 6-bit values into 12 bytes (96 bits = 16 × 6).

### Nibble layout

Each byte in the 128-byte nibble region encodes two 4-bit unsigned values:

```
low_nibble  = byte & 0x0F   → element at even index within sub-block
high_nibble = byte >> 4     → element at odd  index within sub-block
```

Values are **unsigned** [0, 15] — no offset subtraction (unlike Q4_0 which
subtracts 8 to get signed [-8, 7]).

### Dequantisation formula

For element `i` in sub-block `j` (j = i / 32):

```
W[i] = d * scale[j] * q[i]  -  dmin * min[j]
```

where:
- `d`, `dmin` are the f16 superblock factors (decoded to f32)
- `scale[j]`, `min[j]` are the 6-bit sub-block factors (u8, range [0, 63])
- `q[i]` is the raw 4-bit nibble, unsigned [0, 15]

**Output range:** unbounded f32. With typical GGUF files, values fall in
roughly [-10, 10] for attention weights and [-1, 1] for normalisation layers.

---

## Q6_K

### Superblock structure

| Region   | Size      | Description                                       |
|----------|-----------|---------------------------------------------------|
| `ql`     | 128 bytes | Low 4 bits of each 6-bit value, two per byte      |
| `qh`     | 64 bytes  | High 2 bits of each 6-bit value, four per byte    |
| `scales` | 16 bytes  | 16 × i8 sub-block scales (16 sub-blocks of 16)    |
| `d`      | 2 bytes   | f16 — superblock scale factor                     |

**Total: 210 bytes per 256 elements → 6.5625 bits/weight**

### Bit reconstruction

For element `i` (0..255):

```
ql_byte = ql[i / 2]
qh_byte = qh[i / 4]

low4    = (ql_byte >> ((i & 1) * 4)) & 0x0F   // 4 bits from ql
high2   = (qh_byte >> ((i & 3) * 2)) & 0x03   // 2 bits from qh

q6_raw  = low4 | (high2 << 4)                 // unsigned [0, 63]
q       = q6_raw - 32                          // signed   [-32, 31]
```

The `ql` region stores two 4-bit low halves per byte (low nibble = even index,
high nibble = odd index). The `qh` region stores four 2-bit high parts per byte,
two bits per element, packed in order.

### Dequantisation formula

For element `i` in sub-block `j` (j = i / 16):

```
W[i] = d * scales[j] * q[i]
```

where:
- `d` is the f16 superblock scale (decoded to f32)
- `scales[j]` is a signed i8 sub-block scale
- `q[i]` is the signed 6-bit integer, range [-32, 31]

**Output range:** unbounded f32. The signed sub-block scale means W can be
positive or negative regardless of the sign of `q`.

---

## Q5_K

### Superblock structure

| Region   | Size      | Description                                       |
|----------|-----------|---------------------------------------------------|
| `d`      | 2 bytes   | f16 — superblock scale factor                     |
| `dmin`   | 2 bytes   | f16 — superblock min factor                       |
| `scales` | 12 bytes  | Same 6-bit packed format as Q4_K                  |
| `qh`     | 32 bytes  | High (5th) bit for each of the 256 values, 8/byte |
| `ql`     | 128 bytes | Low 4 bits for each of the 256 values, 2/byte     |

**Total: 176 bytes per 256 elements → 5.5 bits/weight**

### Bit reconstruction

Q5_K extends Q4_K by adding one extra bit per value stored in the `qh` region:

```
ql_byte = ql[i / 2]
low4    = (ql_byte >> ((i % 2) * 4)) & 0x0F   // same as Q4_K

qh_byte = qh[i / 8]
bit5    = (qh_byte >> (i % 8)) & 1            // 1 bit per element, 8/byte

q5      = low4 | (bit5 << 4)                  // unsigned [0, 31]
```

The `qh` region packs 8 high bits per byte in natural order (bit 0 = element 0
within the byte group, bit 7 = element 7).

### Dequantisation formula

Identical to Q4_K but with `q5` replacing `q4`:

```
W[i] = d * scale[j] * q5[i]  -  dmin * min[j]
```

where `j = i / 32` and scale/min are decoded from the same 12-byte packed
region as Q4_K.

**Output range:** unbounded f32. The extra bit doubles the quantisation
resolution compared to Q4_K (32 levels vs. 16), reducing reconstruction error
by roughly 50% for the same bits-per-weight budget.

---

## Comparison table

| Format | Bits/weight | Elements/superblock | Sub-blocks | Formula              | q range   |
|--------|-------------|---------------------|------------|----------------------|-----------|
| Q4_K   | 4.5         | 256                 | 8 × 32     | d·sc·q − dmin·mn     | [0, 15]   |
| Q5_K   | 5.5         | 256                 | 8 × 32     | d·sc·q − dmin·mn     | [0, 31]   |
| Q6_K   | 6.5625      | 256                 | 16 × 16    | d·sc·q               | [-32, 31] |

---

## Euler mode interaction

In Euler (cosine projection) mode the sub-block scale/min terms are **not**
applied. The raw integer `q` values are passed directly to `euler_dequant_block`
which normalises them by the per-superblock maximum and projects onto the cosine
curve scaled by `d / φ`. This preserves the GwenTensor output bound
`[-0.618, 0.618]` regardless of the sub-block scale magnitudes.

The rationale is identical to Q8_0/Q4_0 Euler mode: the cosine projection
already encodes relative magnitude through the `d` amplitude term, and adding
an asymmetric per-sub-block offset (the `min` term) would break the symmetry
assumption of the GwenTensor fixed-point accumulator.

---

## Q2_K — 2-bit K-Quant

**Superblock:** 256 elements, 16 sub-blocks × 16 elements.
**Layout:** `[scales: u8×16][qs: u8×64][d: f16][dmin: f16]` = 84 bytes (**2.625 bpw**)

### Scale packing

Each `scales[j]` byte encodes one sub-block's scale index (low nibble) and min
index (high nibble):

```
scale_index[j] = scales[j] & 0x0F   → unsigned [0, 15]
min_index[j]   = scales[j] >> 4     → unsigned [0, 15]
```

### 2-bit extraction

Four 2-bit values are packed per byte, LSB first:

```
q2 = (qs[i / 4] >> ((i % 4) * 2)) & 0x03   → unsigned [0, 3]
```

### Formula

```
W[i] = d × scale_index[j] × q2 − dmin × min_index[j]
```

where `j = i / 16` is the sub-block index.

**Output range:** unbounded f32. With `scale_index ∈ [0,15]`, `q2 ∈ [0,3]`,
and `min_index ∈ [0,15]`, the range before scaling is `[−15, 45]`.

---

## Q3_K — 3-bit K-Quant

**Superblock:** 256 elements, 16 sub-blocks × 16 elements.
**Layout:** `[qs: u8×64][hmask: u8×32][scales: u8×12][d: f16]` = 110 bytes (**3.4375 bpw**)

### 3-bit reconstruction

Each 3-bit value is split across two regions:

```
low2   = (qs[i / 4]    >> ((i % 4) * 2)) & 0x03   // 2 bits from qs
bit2   = (hmask[i / 8] >>  (i % 8))      & 0x01   // 1 bit from hmask
q3_raw = low2 | (bit2 << 2)                        // unsigned [0, 7]
q3     = q3_raw − 4                                // signed   [−4, 3]
```

The `hmask` region packs 8 high bits per byte in natural order (bit 0 = element 0
within the byte group). This is the same pattern as Q5_K's `qh` region.

### Scale packing

The 12-byte `scales` region encodes 16 × 6-bit indices packed **contiguously**
at 6 bits per index = 96 bits = 12 bytes. Extraction spans byte boundaries:

```
bit_pos    = j × 6
byte_index = bit_pos / 8
bit_offset = bit_pos % 8

if bit_offset ≤ 2:
    scale_index[j] = (scales[byte_index] >> bit_offset) & 0x3F
else:
    lo = scales[byte_index]     >> bit_offset
    hi = scales[byte_index + 1] << (8 − bit_offset)
    scale_index[j] = (lo | hi) & 0x3F
```

Each `scale_index[j]` is in `[0, 63]`.

### Formula

```
actual_scale[j] = d × (scale_index[j] / 63.0)
W[i]            = actual_scale[j] × q3
```

Q3_K is **symmetric** — there is no `dmin` term (zero_point = 0).
Dividing by 63.0 normalises the 6-bit index to `[0.0, 1.0]`, so
`actual_scale[j] ∈ [0, d]`.

**Output range:** unbounded f32. With `scale_index = 63` and `d = 1.0`,
output spans `[−4.0, 3.0]`.

---

## Full Comparison Table

| Format | bpw    | Superblock | Sub-blocks | q range  | Formula                        |
|--------|--------|------------|------------|----------|--------------------------------|
| Q2_K   | 2.625  | 256        | 16 × 16    | [0, 3]   | d·s·q − dmin·m                 |
| Q3_K   | 3.4375 | 256        | 16 × 16    | [−4, 3]  | d·(s/63)·q  *(symmetric)*      |
| Q4_K   | 4.5    | 256        | 8 × 32     | [0, 15]  | d·s·q − dmin·m                 |
| Q5_K   | 5.5    | 256        | 8 × 32     | [0, 31]  | d·s·q − dmin·m                 |
| Q6_K   | 6.5625 | 256        | 16 × 16    | [−32, 31]| d·s·q  *(symmetric)*           |
