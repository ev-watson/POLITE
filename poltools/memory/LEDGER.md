# poltools Q&A ledger

Append-only. One atom per answered question. Format:

```
QID-NNN | file::symbol | one-line fact | refs
```

Search this file before re-reading source. Prune never; compress only by human edit.

<!-- Atoms begin below -->

QID-001 | simulate.py::_add_gaussian_psf.stamp_sigma | Half-size of local PSF stamp in units of sigma; default 5.0 truncates Gaussian at ~5σ so >99.999% flux is captured without touching full frame | implementation detail
