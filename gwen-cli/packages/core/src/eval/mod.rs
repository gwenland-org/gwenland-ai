// eval — model evaluation pipeline for GwenLand.
//
// Organised into three concerns so that gwen-tui can call typed functions
// without knowing about sysinfo, reqwest, or JSON serialisation internals:
//
//   metrics     — loss/PPL/throughput/memory measurement (Phase 1)
//   output_eval — inference + substring match scoring (Phase 2)
//   report      — EvalReport struct + JSON serialisation
//
// gwen-tui imports this module and calls the public functions directly;
// it never reaches into the sub-modules.

pub mod metrics;
pub mod output_eval;
pub mod report;
