# Point Processes — Operators

Immutable `equinox.Module` process objects that bundle an intensity model with
sampling and goodness-of-fit diagnostics. Operators are pytrees, safe under
`jit` / `grad` / `vmap`, and delegate all math to the
[primitives](point-process-primitives.md).

::: xtremax.point_processes.operators
    options:
      show_root_heading: false
      show_root_toc_entry: false
