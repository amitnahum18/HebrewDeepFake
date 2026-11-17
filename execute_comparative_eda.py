"""Execute comparison notebook offline, checkpointing outputs on failure."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
path=root/'real_synthetic_comparison.ipynb'
notebook=nbformat.read(path,as_version=4)
def progress(cell,cell_index,**kwargs):
    print('Executing comparison cell',cell_index+1,'of',len(notebook.cells),flush=True)
    nbformat.write(notebook,path)
try:
    NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}},on_cell_start=progress).execute()
finally:
    nbformat.write(notebook,path)
assert not [o for c in notebook.cells for o in c.get('outputs',[]) if o.output_type=='error']
print('Comparison notebook executed successfully')
