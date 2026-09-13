# Gwen Changes - 2026-09-13 - T4 Ceiling Wave 118

- Added an in-process, single-model ABBA stability harness for Wave 111.
- Measured 400 production prefill samples on a verified Tesla T4 with 400/400
  oracle matches.
- Confirmed a small +0.43% median effect (12,729.4 to 12,793.0 tok/s), with a
  bootstrap 95% speedup interval of 0.99908x-1.00814x.
- Closed deferred FFN residual fusion as correct but too small for production
  promotion; it remains opt-in.
- The 15,000 tok/s target remains open.
