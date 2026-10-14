In-situ export to DimensionLab Frontier
=======================================

frontier_emit.py               numpy-only emitter of the `frontier-spool` protocol. Copy it into
                               the solver repository; no Frontier package is needed there.
frontier_catalyst_pipeline.py  ParaView Catalyst 2 pipeline script: writes the same spool from
                               a Catalyst channel (serial), or VTPD files (--mode vtk, parallel).
                               Not validated against ParaView.
adaptor_example.txt            the Conduit node a Catalyst 2 adaptor passes (C++, Fortran,
                               Python), environment variables, other transports.

Reader: `python -m frontier_connectors import catalyst <spool> --out dataset` (plugin `catalyst`).
The protocol is described at the top of frontier_emit.py.
