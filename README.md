# emmy-ptx-compiler

A compiler from Emmy's Kernel IR (`*.ir.json`) straight to NVIDIA PTX.

```
python compile.py path/to/<kernel>.ir.json > out.ptx
```

The IR is loaded with `emmy` (`Graph.from_dict`). Everything after that (the
lowering of every statement, index arithmetic, the math routines and the PTX
text) is done by this project. The launch descriptor `meta.json`, next to the
IR, supplies the parameter order and the launch geometry.

Work in progress.
