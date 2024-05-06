import nbformat
from nbconvert.preprocessors import ExtractOutputPreprocessor
from IPython.display import display
import base64
import os

def extract_images_from_notebook(notebook_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(notebook_path, 'r') as f:
        notebook_content = nbformat.read(f, as_version=4)

    for idx, cell in enumerate(notebook_content['cells'], start=1):
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'data' in output and 'image/png' in output['data']:
                    image_data = output['data']['image/png']
                    image_data_decoded = base64.b64decode(image_data)
                    with open(os.path.join(output_dir, f"output_image_{idx}.png"), "wb") as img_file:
                        img_file.write(image_data_decoded)

if __name__ == "__main__":
    notebook_path = "analisis_grafico.ipynb"
    output_directory = "analisis_grafico_images"
    extract_images_from_notebook(notebook_path, output_directory)
