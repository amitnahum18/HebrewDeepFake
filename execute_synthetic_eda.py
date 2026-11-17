"""Execute the synthetic EDA offline and save outputs, even if a cell fails."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
path=root/'synthetic_eda.ipynb'
notebook=nbformat.read(path,as_version=4)
def progress(cell,cell_index,**kwargs):
    print('Executing synthetic EDA cell',cell_index+1,'of',len(notebook.cells),flush=True)
    nbformat.write(notebook,path)
try:
    NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}},on_cell_start=progress).execute()
finally:
    nbformat.write(notebook,path)
errors=[output for cell in notebook.cells for output in cell.get('outputs',[]) if output.output_type=='error']
assert not errors
print('Synthetic EDA executed with zero cell errors',flush=True)
