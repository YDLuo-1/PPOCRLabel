# Copyright (c) <2015-Present> Tzutalin
# Copyright (C) 2013  MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell, Antonio Torralba,
# William T. Freeman. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
# SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import base64
import ctypes
import logging
import pickle
import os

logger = logging.getLogger("PPOCRLabel")


def _crypt_api_key(data, decrypt=False):
    if os.name != "nt":
        raise OSError("Encrypted API key storage requires Windows; use PPOCRLABEL_API_KEY instead.")

    class DataBlob(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("data", ctypes.c_void_p)]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    local_free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p
    operation = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    operation.argtypes = [
        ctypes.POINTER(DataBlob), ctypes.c_void_p, ctypes.POINTER(DataBlob),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(DataBlob),
    ]
    operation.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(data)
    source = DataBlob(len(data), ctypes.cast(buffer, ctypes.c_void_p))
    result = DataBlob()
    try:
        # UI_FORBIDDEN; omitting LOCAL_MACHINE binds protection to the current user.
        if not operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.memset(buffer, 0, len(data))
        if result.data:
            ctypes.memset(result.data, 0, result.size)
            local_free(result.data)


class Settings(object):
    def __init__(self):
        # Be default, the home will be in the same folder as labelImg
        home = os.path.expanduser("~")
        self.data = {}
        self.path = os.path.join(home, ".autoOCRSettings.pkl")

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default

    def get_api_key(self):
        encrypted = self.get("llm_api_key_dpapi", "")
        if not encrypted:
            return ""
        try:
            return _crypt_api_key(
                base64.b64decode(encrypted, validate=True), decrypt=True
            ).decode("utf-8")
        except (OSError, ValueError, TypeError):
            logger.warning("Cannot decrypt the saved API key; enter it again in automatic labeling settings.")
            return ""

    def set_api_key(self, api_key):
        api_key = api_key.strip()
        encrypted = (
            base64.b64encode(_crypt_api_key(api_key.encode("utf-8"))).decode("ascii")
            if api_key else ""
        )
        self.data.pop("llm_api_key", None)
        if encrypted:
            self.data["llm_api_key_dpapi"] = encrypted
        else:
            self.data.pop("llm_api_key_dpapi", None)

    def save(self):
        if self.path:
            with open(self.path, "wb") as f:
                pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
                return True
        return False

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "rb") as f:
                    self.data = pickle.load(f)
                    return True
        except:
            logger.warning("Loading setting failed")
        return False

    def reset(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            logger.info("Remove setting pkl file %s", self.path)
        self.data = {}
        self.path = None
