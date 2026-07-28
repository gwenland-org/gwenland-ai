//! Which encoding convention a vocabulary uses.
//!
//! Separated from [`super::vocab`] because the choice is not a detail of how a
//! vocabulary is *loaded* — it decides which encoder runs, which decoder runs,
//! and therefore which token ids come out. Two of the three styles share a
//! surface form and differ only in what ranks a merge; see [`Style::SpmBpe`].

/// Which encoding convention a vocabulary uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// SentencePiece-like: `▁` marks spaces, per-token scores drive merges,
    /// unknown bytes fall back to `<0xNN>`.
    Spm,
    /// GPT-2-like: bytes are remapped to printable chars, an explicit merge
    /// list drives merges.
    ByteLevel,
    /// The hybrid Gemma-4 and Sarvam use: SentencePiece *surface form* — `▁`
    /// for spaces, raw UTF-8 rather than the GPT-2 byte remap, `<0xNN>` byte
    /// fallback — but merges driven by an explicit **merge list**, as
    /// byte-level does, not by per-token scores.
    ///
    /// ⚠️ It is worth naming this rather than bending [`Spm`](Style::Spm),
    /// because the two disagree on the thing that decides token ids: SPM will
    /// merge any pair whose concatenation is in the vocabulary, ranked by
    /// score; this style merges only pairs that appear in the merge list,
    /// ranked by position. A vocabulary carrying both (Gemma-4 ships 262 144
    /// scores *and* 514 906 merges) silently produces different ids depending
    /// on which one is believed.
    SpmBpe,
}

/// The SentencePiece space marker.
pub const SPM_SPACE: char = '\u{2581}';

impl Style {
    /// Whether this style writes spaces as [`SPM_SPACE`] and falls back to
    /// `<0xNN>` byte tokens, rather than using the GPT-2 byte remap.
    ///
    /// Shared by [`Spm`](Style::Spm) and [`SpmBpe`](Style::SpmBpe), which is
    /// the whole reason the decoder can treat them as one case while the
    /// encoder must not.
    #[inline]
    pub fn is_sentencepiece_surface(self) -> bool {
        matches!(self, Style::Spm | Style::SpmBpe)
    }
}
