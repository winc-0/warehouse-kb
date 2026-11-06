"""
上传文件名的安全处理。

客户端可以在 multipart 请求里随意填写文件名，FastAPI 会原样交给接口。
如果直接拼进保存路径：
- `../../x.md` 会写到上传目录之外；
- `C:/x.md`、`C:x.md` 这类带盘符的名字会让 `os.path.join` 丢掉前面的目录。
所以保存前只保留最后一段文件名，并拒绝空名、`.`、`..` 和带盘符或控制字符的名字。
"""


class UnsafeFilenameError(ValueError):
    """上传的文件名不能安全地用作本地文件名。"""


def safe_upload_name(filename: str | None) -> str:
    """返回可以安全拼进保存目录的文件名，不安全时抛出 UnsafeFilenameError。"""
    name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if name in ("", ".", ".."):
        raise UnsafeFilenameError(f"文件名无效：{filename!r}")
    if ":" in name or any(ord(ch) < 32 for ch in name):
        raise UnsafeFilenameError(f"文件名含有不允许的字符：{filename!r}")
    return name
