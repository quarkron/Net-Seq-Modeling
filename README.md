# sjkimlab_NETSEQ_TASEP

Original MATLAB code written by Albur Hassan, sjkimlab at University of Illinois at Urbana-Champaign.
Contact: ahassan4@illinois.edu

## Overview

This project simulates how RNAP traffic, ribosome coupling, and Rho-dependent premature termination shape the NET-seq (Nascent Elongating Transcript sequencing) signal along an *E. coli* gene. The simulation uses a TASEP (Totally Asymmetric Simple Exclusion Process) framework.

## Parameters

### Transcription Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RNAPSpeed` | 19 | bp/s | RNAP elongation speed. Based on *E. coli* measurements at 37°C showing ~40-50 nt/s, but can vary by sequence context and growth conditions. The default reflects moderate elongation rates. |
| `kLoading` | 1/20 = 0.05 | s⁻¹ | RNAP loading rate at the promoter. This gives an average inter-arrival time of 20 seconds between RNAPs, creating moderate polymerase traffic. |
| `RNAP_dwellTimeProfile` | ones(geneLength,1) | relative | Position-dependent dwell time weights. A vector of length `geneLength` where higher values indicate slower elongation (pauses). Default uniform profile (all 1s) means constant speed; gene-specific profiles from NET-seq data capture biological pause sites. |
| `RNAP_width` | 35 | bp | RNAP footprint on DNA. Based on structural studies showing RNAP protects ~35 bp of DNA/RNA. This enforces steric exclusion between adjacent RNAPs. |

**Why these values?** RNAP speeds and loading rates are calibrated to match *E. coli* physiology during exponential growth. The position-dependent dwell time profile allows incorporation of sequence-specific pausing observed experimentally.

### Translation Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `ribospeed` | 19 | bp/s | Ribosome translocation speed (~6.3 codons/s, since 3 bp per codon). Bacterial ribosomes typically translate at 15-20 codons/s depending on growth rate and nutrient availability. |
| `kRiboLoading` | 0 (gene-specific) | s⁻¹ | Ribosome loading rate on nascent mRNA. Set to 0 by default but loaded from `Ecoli_gene_TE.csv` based on each gene's translation efficiency in the driver script. |
| `Ribo_width` | 30 | bp | Ribosome footprint on mRNA (~10 codons). Ribosome profiling shows ribosomes protect ~28-30 nt of mRNA. |

**Why these values?** The default ribosome speed matches RNAP speed (both 19 bp/s) to allow coupling dynamics. In reality, ribosomes can move faster, which enables them to "catch up" to paused RNAPs. The ribosome loading rate varies widely by gene (controlled by RBS strength and mRNA structure), so it's loaded from experimental data.

**Co-transcriptional coupling mechanism**: Both RNAP and ribosomes move 5' → 3'. The ribosome loads near the 5' end of the nascent mRNA and follows behind RNAP. When RNAP pauses or slows (according to `RNAP_dwellTimeProfile`), the ribosome catches up from behind and physically collides with it, creating a coupled complex where they move together.

### Rho Termination Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `KRutLoading` | 0.13 | varies | Rho loading parameter. In `PT_Model=2` (default), this scales the probability that Rho loads per unit time based on unprotected RNA length. Higher values = more premature termination. |
| `PT_Model` | 2 | — | Premature termination model. `PT_Model=1`: percentage-based termination at rut sites. `PT_Model=2`: Rho loading probability proportional to exposed nascent RNA length (more biologically realistic). |
| `rutSpeed` | 5 × ribospeed = 95 | bp/s | Rho translocation speed. Rho is an ATP-dependent helicase that moves much faster than ribosomes (~5x faster). If Rho catches RNAP, transcription terminates. |
| `rho_width` | 30 | bp | Rho footprint on RNA. |
| `minRholoadRNA` | 50 | bp | Minimum length of unprotected RNA required for Rho loading. Rho requires a window of exposed RNA to bind. |

**Why these values?** Rho-dependent termination is a critical quality control mechanism in bacteria. Rho loads onto nascent RNA that lacks ribosome protection (e.g., when ribosomes dissociate or don't load efficiently). The fast Rho speed ensures it can catch up to RNAP if there's sufficient exposed RNA. The default `PT_Model=2` captures the biological principle that longer stretches of unprotected RNA increase termination risk.

### Simulation Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `simtime` | 2000 | s | Total simulation duration. Long enough to reach steady-state dynamics and observe nutrient depletion effects. |
| `glutime` | 1600 | s | Glucose phase end time. After this time, **no new RNAPs load** (simulates nutrient depletion/stationary phase entry). Existing RNAPs continue elongating until completion. |
| `dt` | 0.1 | s | Simulation time step for numerical integration. |
| `dx` | 1 | bp | Spatial resolution (1 bp). |
| `geneLength` | 3075 | bp | Gene length in base pairs. Default value is typical for *E. coli* genes (~1 kb coding sequence). Can be overridden by length of `RNAP_dwellTimeProfile`. |

**Why the glucose phase parameter?** This simulates **diauxic shift** or **entry into stationary phase** when glucose (preferred carbon source) is depleted:
- **t = 0–1600 s**: Steady-state transcription with both initiation and elongation (exponential growth).
- **t = 1600–2000 s**: Transcription shutoff—no new initiation, only existing RNAPs finish transcribing. This allows study of NET-seq signal decay and "runoff" dynamics, similar to experimental transcription inhibition assays.

The 400-second runoff window (2000 - 1600) is sufficient for most RNAPs to complete transcription of a 3 kb gene at 19 bp/s (~160 s to traverse the gene).

## Files

### `sjkimlab_NETSEQ_TASEP.m` — Core Simulation Function

A MATLAB function that runs a single stochastic simulation of transcription and co-translational dynamics on one gene. It models:

- **RNAP elongation**: RNAPs load stochastically at the promoter (rate `kLoading`) and step along the gene with exponentially-distributed dwell times. Steric exclusion prevents RNAPs from overlapping (35 bp footprint).
- **Ribosome co-translational coupling**: A ribosome loads onto each mRNA (rate `kRiboLoading`) and translocates along it. When a ribosome catches the RNAP, they become coupled and the RNAP is pushed at the ribosome's speed.
- **Rho-dependent premature termination**: Rho factor can load onto exposed nascent RNA and chase the RNAP. If Rho catches the RNAP, transcription terminates early. Two models are supported:
  - `PT_Model=1`: Percentage-based termination at a rut site.
  - `PT_Model=2` (default): Rho loading probability scales with the length of unprotected RNA behind the RNAP.
- **NET-seq output**: After simulation, it computes a position histogram of all active RNAPs at each integer time point, producing a `NETseq` matrix (position x time).

**Default parameters**: gene length 3075 bp, RNAP speed 19 bp/s, ribosome speed 19 bp/s, simulation time 2000 s, glucose phase ends at 1600 s.

**Usage**:
```matlab
parameters = struct;
parameters.KRutLoading = 0.13;
parameters.RNAP_dwellTimeProfile = ones(1000, 1);
output = sjkimlab_NETSEQ_TASEP(parameters);
```

### `NETSEQ_simulations.m` — Driver Script

A wrapper script that:

1. Specifies a gene of interest (e.g., `'insQ'`).
2. Loads translation efficiency data to set the ribosome loading rate.
3. Loads a gene-specific RNAP dwell-time profile and normalizes it.
4. Runs the TASEP simulation 300 times and averages the NET-seq output.
5. Plots the summed NET-seq signal at time snapshots (t = 200, 400, ..., 1500 s).

**To run**: edit the `gene` variable and `nloci` variable at the top of the script, then execute.

## Input Files

| File | Description |
|------|-------------|
| `Ecoli_gene_TE.csv` | CSV with columns `Gene`, `mRNALevelRPKM`, `TranslationEfficiencyAU`. Provides per-gene ribosome loading rates. |
| `NETSEQ_gene/NETSEQ_<gene>.csv` | Gene-specific CSV containing a numeric vector of RNAP dwell-time weights along the gene, used as the position-dependent transcription speed profile. |

## Output

The script does not write output files. Results are:

- A MATLAB figure plotting summed NET-seq signal (RNAP density vs. gene position).
- Workspace variables (`final_output.NETseq`, `NETseqSum`) available for further analysis.
