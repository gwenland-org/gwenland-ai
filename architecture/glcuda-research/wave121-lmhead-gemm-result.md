# Wave 121 - prefill LM-head GEMM result

## Result

The one-token tensor-core vocabulary projection was correct but not useful.
Six counterbalanced retained/candidate pairs on a Tesla T4 produced 12/12
matching oracle tokens. The candidate improved the isolated LM-head stage by
5.9%, from a 0.5943 ms median to 0.5612 ms, but total GPU prefill regressed:

| measure | retained | candidate |
| --- | ---: | ---: |
| GPU prefill | 19.7877 ms | 19.9827 ms |
| prefill throughput | 12,331.0 tok/s | 12,210.6 tok/s |
| LM head | 0.5943 ms | 0.5612 ms |

Total speedup was 0.9902x. The B-stage duplicate also raised reserved VRAM from
994 MiB to 1,132 MiB. Reject the candidate and remove its runtime path.

## Corrected hotspot picture

Wave 120's first nine-stage run exhausted its event ring because twelve phase
pairs map into only eight per-layer reporting buckets. Wave 121 sized the ring
by actual phase count. A representative retained sample attributes:

- FFN gate/up: 5.0645 ms;
- FFN down: 4.3868 ms;
- attention: 2.9850 ms;
- FFN elementwise: 2.3905 ms;
- QKV: 1.3557 ms;
- attention output: 0.8731 ms;
- LM head: 0.5944 ms.

The earlier apparent 6.5 ms unattributed LM-head hypothesis was therefore an
instrument-capacity artifact, not a production bottleneck. The target remains
15,000 tok/s.

Evidence: `benchmarks/glcuda-t4-wave121-lmhead-gemm-results.zip`, SHA-256
`2400ffa84bd325ab482a763875be501a6cb1200e4b37984166fb79242abdffa6`.
