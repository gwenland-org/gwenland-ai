//! Render: present a session or comparison to a terminal. [`text`] is the
//! default report; [`table`] is the fixed-width table primitive it uses;
//! [`flamegraph`] is an alternate proportional-bar view of the same telemetry.

pub mod flamegraph;
pub mod table;
pub mod text;
