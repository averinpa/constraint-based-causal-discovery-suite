# Doc style options

Four full-page examples of the same content rendered in different
voices. The **substance is identical** (cbcd's PC algorithm — what it
is, how to call it, how to read the output); only the voice, density,
and structure differ. Open each file in your editor and read it as a
page.

| file | voice | inspiration | when it fits |
|---|---|---|---|
| [`style_a_narrative.md`](style_a_narrative.md) | Narrative teaching | scikit-learn, dowhy | Reader learns the *why* before the *how*. Each page is a short essay. |
| [`style_b_terse.md`](style_b_terse.md) | Terse reference | numpy, scipy | Reader knows the theory and wants the signature + an example, fast. |
| [`style_c_conversational.md`](style_c_conversational.md) | Conversational | FastAPI, modern frameworks | Reader is new and wants to feel welcome. Heavy use of admonitions. |
| [`style_d_academic.md`](style_d_academic.md) | Academic | statsmodels, tigramite | Reader will cite the library in a paper. Equations, assumptions, references. |

Same Sphinx + MyST + Furo + Diátaxis machinery underneath all four —
this is purely a **voice and density** decision. Whichever style is
chosen will be applied consistently across cbcd, dagsampler, citk,
bnm so the suite feels like one ecosystem rather than four
independently-written libraries.

## Things to notice while you compare

- **Length.** Same content runs ~50 lines (B), ~60 (C), ~80 (A, D).
  Reading-time tax adds up across a doc set.
- **Where the code sits.** A and C lead with motivation, then code.
  B leads with the signature. D defers code to a "Returns / Examples"
  section after the theory.
- **Tone.** "Let's run it" (A, C) vs. neutral imperative (B) vs. "we"
  / passive (D).
- **What you're paying for.** A: pedagogy. B: scanability. C:
  approachability. D: scholarly weight.
- **What's missing in each.** A: no fast scan; you have to read the
  prose. B: no motivation; you need to know what PC is already.
  C: less precision; admonitions can feel cute. D: intimidating;
  practitioners may bounce.
