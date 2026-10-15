"""Give legacy OpenSees native file I/O ASCII relative paths when possible."""
import os
from pathlib import Path

def install_native_paths(ops):
    if getattr(ops, '_campaign_relative_paths', False): return
    def wrap(fn):
        def call(*args):
            values=list(args)
            for i, value in enumerate(values[:-1]):
                if value in ('-file','-filePath','-fileTime','-binary','-xml') and isinstance(values[i+1], str):
                    path=Path(values[i+1])
                    if path.is_absolute():
                        try: values[i+1]=os.path.relpath(str(path),os.getcwd()).replace('\\','/')
                        except ValueError: pass
            return fn(*values)
        return call
    for name in ('recorder','timeSeries'):
        setattr(ops,name,wrap(getattr(ops,name)))
    ops._campaign_relative_paths=True
