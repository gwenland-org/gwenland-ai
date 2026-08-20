//! Render: present a session or comparison to a terminal. [`text`] is the
//! default report; [`table`] is the fixed-width table primitive it uses;
//! [`flamegraph`] is an alternate proportional-bar view of the same telemetry.

pub mod flamegraph;
/// ASCII loss curve. Ungated — it plots `(step, loss)` pairs and knows nothing
/// about stumman, so the math stays testable in a default build.
pub mod loss_curve;
pub mod table;
pub mod text;
