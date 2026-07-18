# alpyca_tools

Internal Alpaca client helpers and camera operations using the `alpyca` Python library.

This package intentionally does not talk to COM. COM drivers must remain on Windows, accessed via Alpaca.

`fits_writer` is the single POLITE FITS writer. The QHY Alpaca server supplies
pixels via Alpaca/ImageBytes; it is not a FITS producer.
