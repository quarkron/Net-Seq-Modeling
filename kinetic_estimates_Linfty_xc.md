# Kinetics-Only Estimates for \(L_\infty\) and \(x_c\)

This note derives rough biophysical estimates for the plateau exposed-RNA parameters \(L_\infty\) and \(x_c\) using only kinetic parameters from the TASEP model, rather than fitting to the empirical shield factor \(s\).

## 1. Motivation

In the analytic RNAP flux approximation, the exposed nascent RNA length is modeled using a plateau ansatz:

\[
L_{\mathrm{exposed}}(x)
=
L_\infty
\left[
1-\exp\left(-\frac{x-\ell_0}{x_c}\right)
\right].
\]

Here:

- \(L_\infty\) is the effective steady-state exposed RNA length.
- \(x_c\) is the ribosome catch-up or shielding equilibration length scale.
- \(\ell_0\) is the Rho-loading onset position.

The goal is to estimate \(L_\infty\) and \(x_c\) directly from model kinetics, especially the RNAP speed and ribosome loading rate.

## 2. Model parameters

From the TASEP model documentation, the relevant default parameters are:

| Parameter | Symbol | Value | Meaning |
|---|---:|---:|---|
| RNAP speed | \(v_{\mathrm{RNAP}}\) | \(19\ \mathrm{bp/s}\) | Average RNAP elongation speed |
| Ribosome speed | \(v_{\mathrm{ribo}}\) | \(19\ \mathrm{bp/s}\) | Average ribosome translocation speed |
| RNAP footprint | \(w_{\mathrm{RNAP}}\) | \(35\ \mathrm{bp}\) | Excluded footprint of RNAP |
| Ribosome offset / footprint term | — | \(30\ \mathrm{bp}\) | Approximate ribosome-associated protected spacing |
| Minimum Rho loading window | \(L_{\min}\) | \(50\ \mathrm{bp}\) | Minimum unprotected RNA needed for Rho loading |
| Ribosome loading rate | \(k_{\mathrm{ribo}}\) | gene-specific | Translation initiation / ribosome loading rate |

The exposed RNA length used by the Rho model is approximately

\[
L_{\mathrm{exposed}}
=
x_{\mathrm{RNAP}} - w_{\mathrm{RNAP}} - 30 - x_{\mathrm{ribo}}.
\]

Since

\[
w_{\mathrm{RNAP}} + 30 = 35 + 30 = 65\ \mathrm{bp},
\]

and Rho loading requires at least \(50\ \mathrm{bp}\) of exposed RNA, the onset position is

\[
\ell_0 = 65 + 50 = 115\ \mathrm{bp}.
\]

## 3. Estimate for \(x_c\)

Assume ribosome loading is a Poisson process with rate \(k_{\mathrm{ribo}}\). Then the mean waiting time for the first ribosome to load is

\[
\tau_{\mathrm{load}} \approx \frac{1}{k_{\mathrm{ribo}}}.
\]

During this time, RNAP moves downstream by approximately

\[
v_{\mathrm{RNAP}} \tau_{\mathrm{load}}
=
\frac{v_{\mathrm{RNAP}}}{k_{\mathrm{ribo}}}.
\]

Therefore, a natural kinetics-only estimate for the catch-up or shielding equilibration length is

\[
\boxed{
x_c^{(0)} \sim \frac{v_{\mathrm{RNAP}}}{k_{\mathrm{ribo}}}
}
\]

or, using \(v_{\mathrm{RNAP}}=19\ \mathrm{bp/s}\),

\[
\boxed{
x_c^{(0)} \sim \frac{19}{k_{\mathrm{ribo}}}\ \mathrm{bp}.
}
\]

This is the distance RNAP travels, on average, before a ribosome loads and begins shielding the nascent transcript.

## 4. Estimate for \(L_\infty\)

A crude estimate for the steady-state exposed RNA length is the distance RNAP travels before ribosome loading, minus the protected offset:

\[
L_\infty^{(0)}
\sim
\frac{v_{\mathrm{RNAP}}}{k_{\mathrm{ribo}}}
-
(w_{\mathrm{RNAP}}+30).
\]

However, the Rho model only allows Rho loading once the exposed RNA exceeds the minimum loading window \(L_{\min}=50\ \mathrm{bp}\). Therefore, we impose a lower bound:

\[
\boxed{
L_\infty^{(0)}
\sim
\max\left[
L_{\min},
\frac{v_{\mathrm{RNAP}}}{k_{\mathrm{ribo}}}
-
(w_{\mathrm{RNAP}}+30)
\right].
}
\]

Using the default values \(v_{\mathrm{RNAP}}=19\ \mathrm{bp/s}\), \(w_{\mathrm{RNAP}}=35\ \mathrm{bp}\), and \(L_{\min}=50\ \mathrm{bp}\), this becomes

\[
\boxed{
L_\infty^{(0)}
=
\max\left[
50,
\frac{19}{k_{\mathrm{ribo}}}-65
\right]\ \mathrm{bp}.
}
\]

This estimate says that weakly translated genes can have large exposed RNA lengths, while strongly translated genes are clamped close to the minimum Rho-accessible window.

## 5. Example estimates for five genes

Using the ribosome loading rates from the five-gene panel:

| Gene | \(k_{\mathrm{ribo}}\) \((\mathrm{s}^{-1})\) | \(x_c^{(0)} = 19/k_{\mathrm{ribo}}\) bp | \(19/k_{\mathrm{ribo}} - 65\) bp | \(L_\infty^{(0)}\) bp |
|---|---:|---:|---:|---:|
| insQ | 0.022 | 864 | 799 | 799 |
| talB | 0.506 | 38 | -27 | 50 |
| aceA | 0.358 | 53 | -12 | 50 |
| dnaK | 0.456 | 42 | -23 | 50 |
| rpoB | 0.166 | 114 | 49 | 50 |

Under this kinetics-only approximation, most moderately or strongly translated genes have \(L_\infty\) near the Rho-loading minimum, while very weakly translated genes such as `insQ` have much larger exposed RNA.

## 6. Interpretation

The estimates can be summarized as:

\[
\boxed{
x_c^{(0)} = \frac{v_{\mathrm{RNAP}}}{k_{\mathrm{ribo}}}
}
\]

and

\[
\boxed{
L_\infty^{(0)}
=
\max\left[
L_{\min},
\frac{v_{\mathrm{RNAP}}}{k_{\mathrm{ribo}}}-(w_{\mathrm{RNAP}}+30)
\right].
}
\]

Biologically:

- Larger \(x_c\) means ribosome shielding takes longer to establish downstream of the promoter.
- Larger \(L_\infty\) means more nascent RNA remains exposed to Rho at steady state.
- Low \(k_{\mathrm{ribo}}\) produces both large \(x_c\) and large \(L_\infty\).
- High \(k_{\mathrm{ribo}}\) causes rapid ribosome shielding and clamps \(L_\infty\) near the minimum Rho-loading window.

## 7. Caveats

These are zeroth-order, kinetics-only estimates. They should be used as priors or initialization guesses, not as final fitted values.

They ignore:

1. Position-dependent RNAP pausing through \(D(x)\).
2. Ribosome catch-up caused by RNAP pauses.
3. Differences between RNAP and ribosome elongation speeds.
4. RNAP traffic and hard-body exclusion.
5. Failed or delayed translation initiation beyond the simple Poisson loading assumption.
6. Gene-specific coupling dynamics.

A more complete estimate would fit \(L_\infty\), \(x_c\), and the traffic parameter \(a\) directly to the simulated flux profile \(j_{\mathrm{sim}}(x)\). These kinetics-only values provide physically motivated starting points for that fit.

## 8. Optional refinement if ribosomes move faster than RNAP

If ribosomes move faster than RNAP, \(v_{\mathrm{ribo}}>v_{\mathrm{RNAP}}\), then one can include a deterministic catch-up time. A rough estimate is

\[
t_{\mathrm{catch}}
\approx
\frac{v_{\mathrm{RNAP}}/k_{\mathrm{ribo}}}{v_{\mathrm{ribo}}-v_{\mathrm{RNAP}}}.
\]

Then the equilibration length could be approximated as

\[
x_c
\sim
v_{\mathrm{RNAP}}
\left(
\frac{1}{k_{\mathrm{ribo}}}
+
t_{\mathrm{catch}}
\right).
\]

However, in the default model,

\[
v_{\mathrm{ribo}} = v_{\mathrm{RNAP}} = 19\ \mathrm{bp/s},
\]

so deterministic catch-up by speed alone does not occur. Ribosome catch-up then mainly arises when RNAP pauses or slows through the dwell-time profile \(D(x)\).

## 9. Addendum: per-RNAP physics and the saturating-plateau correction

The estimates in sections 3–4 implicitly treat ribosomes as a "frontmost ribosome among many" that establishes a steady-state lag set by the inverse loading rate. The actual kernel uses a different model, which changes both the derivation and the predicted asymptotic behavior.

### 9.1 Per-RNAP, single-ribosome model in the kernel

Both the MATLAB reference (`sjkimlab_NETSEQ_TASEP.m`, lines 213/334/341–352) and the Python ports (`netseq_tasep_fast.py`, `netseq_tasep_gpu.py`) implement a one-ribosome-per-RNAP model:

- When an RNAP loads at the promoter, a ribosome is scheduled with delay \(t \sim \mathrm{Exp}(1/k_{\mathrm{ribo}})\).
- The ribosome loads at position 1 once the RNAP has reached position 30 (the ribosome footprint).
- Both move at \(v_{\mathrm{ribo}} = v_{\mathrm{RNAP}} = 19\) bp/s, so once loaded, the gap \(g = v_{\mathrm{RNAP}}\,t\) stays constant.
- The MATLAB header is explicit: *"this function does not include many features such as RNA degradation multiple ribosomes."*

In particular, there is no "frontmost ribosome among many" — each RNAP runs its own race against Rho on its own mRNA, with its own assigned ribosome whose lag is fixed at the loading event.

### 9.2 Closed-form de novo survival

For an RNAP at position \(x\) with frozen gap \(g\), the unprotected RNA profile seen by Rho is:

\[
L_{\mathrm{unc}}(x'; g) =
\begin{cases}
0 & x' \le \ell_0 \\
x' - \ell_0 & \ell_0 < x' \le g \quad \text{(ribosome not loaded yet)} \\
g - \ell_0 & x' > g \quad \text{(ribosome present, frozen gap)}
\end{cases}
\]

with \(\ell_0 = w_{\mathrm{RNAP}} + 30 = 65\) bp. If \(g \le \ell_0\) the ribosome is always within the protected zone and Rho cannot load anywhere along the gene.

Substituting this single-RNAP \(L_{\mathrm{unc}}\) into the M8 hazard form gives the per-RNAP survival as a function of the loading delay \(t\):

\[
\Pi(x; t) = \exp\!\left[-\frac{K_{\mathrm{rut}}}{v_{\mathrm{RNAP}}\,L_{\mathrm{gene}}}
\int_0^x L_{\mathrm{unc}}(x'; v_{\mathrm{RNAP}}\,t)\, \tilde D(x')\,\mathrm{d}x'\right].
\]

The population-averaged survival, with \(t\) drawn from the loading-delay distribution, is

\[
\boxed{
\langle \Pi(x)\rangle
=
\int_0^\infty \Pi(x; t)\, k_{\mathrm{ribo}}\,e^{-k_{\mathrm{ribo}}\,t}\,\mathrm{d}t.
}
\tag{M13}
\]

This is **closed form**: no parameters are fit. Inputs are \(k_{\mathrm{ribo}}\), \(K_{\mathrm{rut}}\), and a dwell-time profile \(\tilde D\) (which can be the CMA-ES \(D^*\), the experimental \(S_{\mathrm{exp,norm}}\) as a proxy, or simply 1).

### 9.3 Two scalar limits

**Full asymptote (M13a).** Take the late-\(x\) limit with \(\langle \tilde D\rangle = 1\). The cumulative integrand at \(x = L_{\mathrm{gene}}\) decomposes into a no-ribosome part on \([\ell_0, g]\) plus a constant-\(L_{\mathrm{unc}}\) part on \([g, L_{\mathrm{gene}}]\):

\[
\int_0^{L_{\mathrm{gene}}} L_{\mathrm{unc}}(x'; g)\,\mathrm{d}x'
=
\tfrac{1}{2}(g-\ell_0)^2 + (g - \ell_0)(L_{\mathrm{gene}} - g)
\;\approx\;
(g - \ell_0)\, L_{\mathrm{gene}}
\quad \text{for } g \ll L_{\mathrm{gene}}.
\]

Substituting and averaging over \(g \sim \mathrm{Exp}(k_{\mathrm{ribo}}/v_{\mathrm{RNAP}})\):

\[
\boxed{
\langle \Pi(L_{\mathrm{gene}})\rangle
\;\approx\;
\bigl(1 - e^{-\ell_0\,k_{\mathrm{ribo}}/v_{\mathrm{RNAP}}}\bigr)
\;+\;
e^{-\ell_0\,k_{\mathrm{ribo}}/v_{\mathrm{RNAP}}} \cdot
\frac{k_{\mathrm{ribo}}}{k_{\mathrm{ribo}} + K_{\mathrm{rut}}}.
}
\tag{M13a}
\]

The first term is the probability the ribosome loaded before the RNAP cleared its protected zone — those RNAPs are *fully* shielded. The second term is the conditional survival of the unshielded fraction.

**Naked-race (M14).** Dropping the protected-zone correction (\(\ell_0 \to 0\)) gives

\[
\boxed{
\langle \Pi(L_{\mathrm{gene}})\rangle^{(M14)}
\;\approx\;
\frac{k_{\mathrm{ribo}}}{k_{\mathrm{ribo}} + K_{\mathrm{rut}}}.
}
\tag{M14}
\]

This is the simplest possible scalar prediction — a one-line race between two Poisson processes (ribosome loading vs. Rho loading), no geometry. It is *not* the rigorous limit of M13; it is a heuristic that happens to be empirically useful (see 9.4).

Both M13a and M14 saturate at a **non-zero** value as \(L_{\mathrm{gene}} \to \infty\), qualitatively unlike the M9 plateau ansatz of section 6 which predicts \(\Pi \to 0\) exponentially. The two functional forms can fit each other to small log-MSE over a finite gene length, but they extrapolate differently and the M9 \((L_\infty, x_*)\) parameters lose their kinetic interpretation.

### 9.4 Empirical check on the 5-gene panel

Comparing all three predictions to the kernel's `flux_norm[L]/flux_norm[1]`:

| gene | \(L\) | \(k_{\mathrm{ribo}}\) | observed \(j_{n0}[L]\) | M13a (rigorous) | M14 (naked race) | M13 numerical (\(\tilde D = S_{\mathrm{exp,norm}}\)) |
|------|---:|---:|---:|---:|---:|---:|
| insQ | 1149 | 0.022 | **0.158** | 0.208 | 0.145 | 0.249 |
| talB |  954 | 0.506 | **0.517** | 0.964 | 0.796 | 0.969 |
| aceA | 1305 | 0.358 | **0.654** | 0.922 | 0.734 | 0.927 |
| dnaK | 1917 | 0.456 | **0.433** | 0.953 | 0.778 | 0.957 |
| rpoB | 4029 | 0.166 | **0.500** | 0.751 | 0.561 | 0.758 |

Two findings stand out:

1. **All three predictions over-shoot the observed plateau on every gene.** The rigorous M13a is the *worst* (over-shoots by 30–120%); the heuristic M14 is the closest (within 9–12% on insQ/aceA/rpoB) but still over-predicts on talB/dnaK by 54–80%. The numerical M13 with the experimental \(\tilde D\) tracks M13a closely — pause structure does not reduce survival via this integrand.

2. **The kernel does saturate.** Late-quartile log-slope is \(\le 3\times 10^{-4}\) bp\(^{-1}\) for `aceA`, `dnaK`, `rpoB` — confirming the qualitative per-RNAP picture (finite plateau, not exponential decay to zero). The functional form is right; the magnitude is off.

### 9.5 Reading the discrepancy --- the kernel's snap-to-negative artifact

Tracing the systematic over-prediction back to the kernel uncovered a real per-RNAP behavior that M13 of section 9.2 does not capture. The ribosome elongation block (`netseq_tasep_fast.py` line 595, identical to `sjkimlab_NETSEQ_TASEP.m` line 561) snaps the ribosome to position \(x_{\mathrm{RNAP}} - w_{\mathrm{RNAP}}\) on collision detection. When the ribosome loads at position 1 with the RNAP at the loading-gate minimum \(x_{\mathrm{RNAP}} = 30\), the very next elongation step's overlap test fires and snaps the ribosome to \(30 - 35 = -5\). The subsequent ribosome elongation block is gated by \(\texttt{Ribo\_locs} > 0\), so the ribosome remains permanently stuck at the negative position --- the Rho-loading test reads the negative \(\texttt{Ribo\_locs}\) directly into

\[
\texttt{PTRNAsize} = x_{\mathrm{RNAP}} - 65 - \texttt{Ribo\_locs},
\]

inflating the exposed-RNA length and effectively giving the affected RNAPs no shielding for the rest of their transit. The artifact is in the canonical sjkimlab MATLAB; see also \texttt{bcm\_sweep\_observations.tex} \S "Known kernel artifact."

The snap fires whenever the RNAP position at the first elongation step after loading is below \(\sim w_{\mathrm{RNAP}} + \langle\texttt{tempRibo}\rangle = 35 + 2 \approx 37\) bp. The fraction of RNAPs falling in this regime is

\[
P_{\mathrm{snap}} \;\approx\; 1 - \exp\!\left(-\frac{w_{\mathrm{RNAP}}\,k_{\mathrm{ribo}}}{v_{\mathrm{RNAP}}}\right),
\]

which for the panel ranges from 4\% (insQ, slow loading) to 60\% (talB, fast loading).

### 9.6 Refined closed form: M13 with snap correction

Splitting the population into snap-fired and snap-avoiding subpopulations gives a refined zero-parameter formula:

\[
\boxed{
\langle \Pi(x)\rangle
=
P_{\mathrm{snap}} \cdot \langle\Pi_{\mathrm{no\text{-}ribo}}(x)\rangle
\;+\;
(1 - P_{\mathrm{snap}}) \cdot \langle\Pi_{\mathrm{M13}}(x)\,\big|\,t > w_{\mathrm{RNAP}}/v_{\mathrm{RNAP}}\rangle.
}
\tag{M15}
\]

The two branches are:

- **No-ribosome branch.** For snap-fired RNAPs, \(\texttt{Ribo\_locs}\) is permanently \(\sim -5\) and the unprotected RNA length grows with the RNAP position:
\[
\langle\Pi_{\mathrm{no\text{-}ribo}}(x)\rangle
=
\exp\!\left(-\frac{K_{\mathrm{rut}}}{v_{\mathrm{RNAP}}\,L_{\mathrm{gene}}}\int_{\ell_0+\ell_{\min}}^{x}(x'-\ell_0)\,\tilde D(x')\,\mathrm{d}x'\right),
\]
where \(\ell_{\min} = 50\) is the kernel's minimum exposed-RNA threshold for Rho loading.

- **Per-RNAP shielded branch.** For snap-avoiding RNAPs, the loading delay is restricted to \(t > w_{\mathrm{RNAP}}/v_{\mathrm{RNAP}}\) and the M13 integrand applies as derived in \S 9.2, with the additional gate that \(L_{\mathrm{unc}} > \ell_{\min}\) is required for hazard to fire (so the post-loading plateau contributes only when \(g > \ell_0 + \ell_{\min}\)).

### 9.7 Empirical check on the 5-gene panel

| gene | \(L\) | \(k_{\mathrm{ribo}}\) | observed \(j_{n0}[L]\) | M13 (orig) | M15 (snap-aware) | residual |
|------|---:|---:|---:|---:|---:|---:|
| insQ | 1149 | 0.022 | 0.158 | 0.249 | 0.220 | $+6.2\%$ |
| talB |  954 | 0.506 | 0.517 | 0.969 | 0.438 | $-7.9\%$ |
| aceA | 1305 | 0.358 | 0.654 | 0.927 | 0.477 | $-17.7\%$ |
| dnaK | 1917 | 0.456 | 0.433 | 0.957 | 0.406 | $-2.7\%$ |
| rpoB | 4029 | 0.166 | 0.500 | 0.758 | 0.522 | $+2.3\%$ |

Mean absolute error drops from $32\%$ (M13) to $7\%$ (M15). Three of five panel genes (`insQ`, `dnaK`, `rpoB`) are within 6\%; `talB` and `aceA` slightly under-shoot, indicating the snap-fired branch is treated as more "no-ribo" than it actually is (the snap-stuck ribosome at \(-5\) shifts the effective Rho-loading threshold from \(x_{\mathrm{RNAP}} > 115\) to \(x_{\mathrm{RNAP}} > 110\), a small correction we ignore for clarity). All predictions have **zero free parameters** --- inputs are the kinetic constants \(k_{\mathrm{ribo}}\), \(K_{\mathrm{rut}}\), and either the experimental \(S_{\mathrm{exp,norm}}\) or the optimized \(D^*\) as the dwell profile \(\tilde D\).

### 9.8 Reading

- **For the patched kernel** (Python kernels patched per OpenSpec change `ribo-snap-bias-diagnostic`), M13 of \S 9.2 is the right closed form. The diagnostic notebook `ribo_bias.ipynb` confirms M13 matches the patched-kernel flux endpoint within 5\% on 3 of 5 panel genes (`talB`: 2.7\%, `aceA`: 3.1\%, `dnaK`: 3.5\%).
- **For the unpatched kernel** (the canonical sjkimlab MATLAB and any pre-patch Python output), M15 of \S 9.6 is the right closed form. It modifies M13 only by pre-multiplying the no-ribosome and per-RNAP branches by their respective population fractions \((P_{\mathrm{snap}},\, 1 - P_{\mathrm{snap}})\); the integrand structure is unchanged.
- **The snap probability \(P_{\mathrm{snap}} = 1 - \exp(-w_{\mathrm{RNAP}} k_{\mathrm{ribo}} / v_{\mathrm{RNAP}})\)** is the single piece of new physics: it captures the fraction of RNAPs whose ribosome loaded too close to the RNAP for the first overlap-snap event to leave the ribosome in a positive position.
- **The point-particle plug-ins of \S\S 3--4** remain useful as zeroth-order priors but have been superseded by M13 / M15 as the quantitative closed form.

The notebook `mathmodel-netseq.ipynb` overlays M15 against the M9 fit and the kernel `j_n0` for the panel, with M13 (orig) shown as a dashed reference to make the snap correction visible.

### 9.6 Reading

- The point-particle plug-ins of sections 3–4 (\(L_\infty^{(0)},\,x_c^{(0)}\)) remain useful as zeroth-order priors for \((L_\infty, x_*)\) in fits, but their interpretation as "steady-state ribosome lag" does not survive the per-RNAP physics of section 9.1.
- The closed-form M13 is the natural next refinement: it extends the same kinetic inputs to a position-dependent survival prediction with **zero free parameters**.
- The scalar plateau M14 is the simplest diagnostic: it nails 3/5 panel genes and flags the remaining two as targets for further refinement.
