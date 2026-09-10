# TalorDet / AnchorTaylor

This repository contains the development history and reproducible release for
**AnchorTaylor: Single-Scale High-Order Fusion for Efficient RGB-Infrared
Object Detection**.

The clean release is under [`talordet_release/`](talordet_release/). It contains:

- the modified Ultralytics 8.1.47 code used by the paper;
- pretrained RGB anchoring with paired RGB/infrared input;
- P2 budgeted bidirectional detail exchange;
- P3 third-order recursive correction;
- dataset templates, three-seed scripts, environment capture, and result aggregation;
- no datasets, experiment runs, or trained checkpoints.

Start with [`talordet_release/README.md`](talordet_release/README.md). Historical
research files at the repository root are retained for traceability and are not
the recommended reproduction entry point.

## License

The bundled Ultralytics-derived code is distributed under AGPL-3.0. See
[`talordet_release/LICENSE`](talordet_release/LICENSE) and
[`talordet_release/THIRD_PARTY_NOTICES.md`](talordet_release/THIRD_PARTY_NOTICES.md).
