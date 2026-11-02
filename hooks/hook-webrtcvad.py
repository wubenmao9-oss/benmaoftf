"""覆盖 PyInstaller 自带的 hook-webrtcvad。

webrtcvad-wheels 的包结构和官方 webrtcvad 不同，自带 hook 在导入探针模块时会失败，
这里改成按文件收集。
"""

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("webrtcvad")
