# tan_gui.spec
import os
from PyInstaller.utils.hooks import collect_all

# Collect all torch data, binaries, and hidden imports
datas = []
binaries = []
hiddenimports = []

# Collect torch (critical for PyTorch apps)
tmp_ret = collect_all('torch')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collect rlcard if it's a package
try:
    tmp_ret = collect_all('rlcard')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except Exception as e:
    print(f"Warning: Could not collect rlcard: {e}")

# Add your model files explicitly
# Format: ('source/path/on/disk', 'destination/path/in/bundle')
model_files = [
    ('experiments/tan_dmc/tan/0_2081369600.pth', 'experiments/tan_dmc/tan'),
    ('experiments/tan_dmc/tan/1_2081369600.pth', 'experiments/tan_dmc/tan'),
    ('experiments/tan_dmc/tan/2_2081369600.pth', 'experiments/tan_dmc/tan'),
]

for src, dest in model_files:
    if os.path.exists(src):
        datas.append((src, dest))
    else:
        print(f"Warning: Model file not found: {src}")

# Add any other data files your app needs
datas.append(('tan_common.py', '.'))
datas.append(('tan_rlcard.py', '.'))
datas.append(('tan_game.py', '.'))

a = Analysis(
    ['tan_gui.py'],  # Your main entry script
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary large packages to reduce size
        'matplotlib', 'scipy', 'pandas', 'IPython',
        'notebook', 'jupyter', 'pytest',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TanGame',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True if you want to see terminal output
    disable_windowed_traceback=False,
    argv_emulation=False,  # True causes re-spawn loops on newer macOS
    target_arch=None,    # Set to 'arm64' or 'x86_64' if needed
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TanGame',
)

# For macOS .app bundle, add BUNDLE
app = BUNDLE(
    coll,
    name='TanGame.app',
    icon=None,  # Add path to .icns file if you have one
    bundle_identifier='com.yourname.tangame',
    info_plist={
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
    },
)