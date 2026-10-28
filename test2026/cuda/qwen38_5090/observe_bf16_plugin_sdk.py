"""Read-only installed TensorRT identity before any wrapper executable launch."""
from pathlib import Path
import datetime,hashlib,json,subprocess
root=Path('D:/CantorAI/5090-study-20261023');dll=root/'tools/TensorRT/bin/nvinfer_10.dll'
destination=root/'results/bf16-plugin-sdk-runtime-observation.json';assert not destination.exists()
command=f"(Get-Item -LiteralPath '{dll}').VersionInfo | Select-Object FileVersion,ProductVersion | ConvertTo-Json -Compress"
version=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],text=True,encoding='utf8'))
record={'path':str(dll),'bytes':dll.stat().st_size,'sha256':hashlib.sha256(dll.read_bytes()).hexdigest(),
        'version':version,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Read-only byte hash and Windows file version of installed study SDK. No DLL loaded/replaced, no benchmark or wrapper execution.'}
destination.write_text(json.dumps(record,indent=2),encoding='utf8');print(json.dumps(record,indent=2))
