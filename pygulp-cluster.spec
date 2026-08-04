# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


ase_spacegroup_datas = collect_data_files('ase.spacegroup')

a = Analysis(
    ['src/pygulp/cli.py'],
    pathex=['src'],
    binaries=[],
    datas=ase_spacegroup_datas,
    hiddenimports=['ase.io.cif', 'ase.io.extxyz', 'openpyxl', 'yaml'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Use the cluster's system libgcc_s; the Conda copy may require a newer glibc.
a.binaries = [
    item
    for item in a.binaries
    if item[0] != 'libgcc_s.so.1' and Path(item[1]).name != 'libgcc_s.so.1'
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pygulp-cluster',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
