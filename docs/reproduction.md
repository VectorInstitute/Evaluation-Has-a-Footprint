# Reproduction

The portable runner reproduces the finalized two-model Qwen evaluation protocol
from caller-prepared frozen data. It never downloads benchmark data, schedules
jobs, or configures infrastructure.

Install lightweight validation and documentation dependencies with `uv sync`.
For inference, install `uv sync --group campaign-reproduction`. The accepted
campaign used Python 3.11.15, PyTorch 2.11.0+cu130, Transformers 4.57.6,
Accelerate 1.14.0, bitsandbytes 0.50.1, qwen-vl-utils 0.0.14, torchvision
0.26.0+cu130, and Pillow 12.3.0. CUDA-tagged PyTorch wheels must be installed
from the appropriate PyTorch CUDA 13.0 index; the normal portable dependency
group intentionally does not claim numerical equivalence with another build.

Prepared data must be supplied separately as a directory containing `sample.csv`
and its matching `sampling.json`; prepared BBQ-V also requires an `images/`
directory with the hash-verified JPEG files. Invoke the runner with a pinned
model, dataset, condition, prepared-data directory, output directory, and the
explicit frozen membership file for M4/M5. The metadata-only manifests in
`data/manifests/` reproduce the paper's retained units exactly, while
`evaluation_has_a_footprint.subsets` deterministically regenerates and validates
them against separately prepared upstream data. Gemma is not included in the
current public scope.

Telemetry is optional: install it with `uv sync --group telemetry` in addition
to the inference dependencies. Use `--telemetry` to measure only prediction
generation. The explicit `--carbon-profile accepted-campaign` reproduces the
59.0 gCO2e/kWh Ontario/ECCC, location-unconfirmed campaign scenario; for other
settings, provide `--carbon-intensity-g-per-kwh` rather than assuming a
location. `footprint.json` records NVML GPU-attributed energy as the primary
measurement and CodeCarbon as a broader secondary cross-check. Water is always
reported as an estimated 1.8–4.0 L/kWh range, with CodeCarbon total energy as
the primary basis and NVML GPU energy as a secondary basis.

The public parser reproduces the finalized general Qwen parsing and deterministic
maintenance-recovery methodology. Two historical Qwen2.5 BBQ-V INT4 records
required artifact-specific recorded policy resolutions; those decisions are not
generalized here. This release is therefore not a byte-for-byte recreation
mechanism for every historical accepted prediction artifact.
