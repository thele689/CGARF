"""Compatibility shim for Python builds without the optional _lzma module.

Some local Python 3.10 builds omit ``_lzma``. The official SWE-bench harness
imports HuggingFace ``datasets``, which imports ``lzma`` unconditionally even
when evaluating a local JSON dataset. CGARF's harness runner prepends this
directory to ``PYTHONPATH`` only for that subprocess.

The shim intentionally raises when actual LZMA compression/decompression is
requested. It is safe for the local JSON SWE-bench path used by CGARF.
"""


class LZMAError(Exception):
    pass


FORMAT_AUTO = 0
FORMAT_XZ = 1
FORMAT_ALONE = 2
FORMAT_RAW = 3
CHECK_NONE = 0
CHECK_CRC32 = 1
CHECK_CRC64 = 4
CHECK_SHA256 = 10
CHECK_ID_MAX = 15
CHECK_UNKNOWN = 16
FILTER_LZMA1 = 0x4000000000000001
FILTER_LZMA2 = 0x21
FILTER_DELTA = 0x03
FILTER_X86 = 0x04
FILTER_IA64 = 0x05
FILTER_ARM = 0x07
FILTER_ARMTHUMB = 0x08
FILTER_POWERPC = 0x05
FILTER_SPARC = 0x09
MF_HC3 = 0x03
MF_HC4 = 0x04
MF_BT2 = 0x12
MF_BT3 = 0x13
MF_BT4 = 0x14
MODE_FAST = 1
MODE_NORMAL = 2
PRESET_DEFAULT = 6
PRESET_EXTREME = 1 << 31


def _unavailable(*_args, **_kwargs):
    raise LZMAError("This Python build has no _lzma support")


open = _unavailable
compress = _unavailable
decompress = _unavailable
is_check_supported = lambda _check: False


class LZMAFile:
    def __init__(self, *_args, **_kwargs):
        _unavailable()
