"""Execute and persist the notebook, including partial outputs on failure."""
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
path=root/'dataset.ipynb'
nb=nbformat.read(path,as_version=4)
def progress(cell,cell_index,**kwargs):
    print(f'Executing cell {cell_index+1}/{len(nb.cells)}',flush=True)
    nbformat.write(nb,path)
class ProgressNotebookClient(NotebookClient):
    def process_message(self,msg,cell,cell_index):
        if msg['msg_type']=='stream' and 'Decoded' in msg['content'].get('text',''):
            print(msg['content']['text'],end='',flush=True)
        return super().process_message(msg,cell,cell_index)
client=ProgressNotebookClient(nb,timeout=3600,kernel_name='python3',resources={'metadata':{'path':str(root)}},on_cell_start=progress)
started=time.monotonic()
try:
    client.execute()
finally:
    nbformat.write(nb,path)
errors=[(i,o) for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.output_type=='error']
executed=[c for c in nb.cells if c.cell_type=='code' and c.execution_count is not None]
print(f'Completed {len(executed)} code cells, {len(errors)} errors, elapsed {(time.monotonic()-started)/60:.1f} minutes',flush=True)
if errors: raise RuntimeError(errors)
