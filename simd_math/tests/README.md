# simd_math tests

Per-primitive error-profile sweeps that derive max relative/absolute error

## Running

```
pixi run test
```

Individual tests can also be run directly:

```
pixi run test-exp
pixi run test-log
pixi run test-sincos
```

## libc / libm dependency

Sincos test requires libc only for validation.
