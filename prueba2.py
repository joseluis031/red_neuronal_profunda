from nbconvert import NotebookExporter
from nbformat import read

def execute_notebook_cell(notebook_path, cell_index):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = read(f, as_version=4)

    exporter = NotebookExporter()
    body, _ = exporter.from_notebook_node(notebook)

    cells = notebook['cells']
    cell = cells[cell_index]

    if cell['cell_type'] == 'code':
        exec(cell['source'])
if __name__ == "__main__":
    notebook_path = "ARIMA_comparar/sarima_y_arima_comparacion.ipynb"
    cell_index = 10 # Índice de la celda que deseas ejecutar
    execute_notebook_cell(notebook_path, cell_index)
