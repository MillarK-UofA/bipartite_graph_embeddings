# encoding: utf-8
# module generate_docs.py
# from docs

"""
This module automates the process for creating mardown (md) files from the python modules in a specified directory.
It formats the doc strings from each file to

How To:
    1. Navigate to the root of the bipartite-graph-embeddings code base.
    2. run "python3 ./docs/automacdoc/generate_docs.py"
    3. Navigate to ./docs
    4. run "python3 -m mkdocs serve"
    5. navigate to "http://localhost:8000/" on your browser.
"""

# - Change working directory to root of bipartite-graph-embeddings --------------------------------------------------- #
import os
from pathlib import Path

ROOT_DIR = Path(os.path.realpath(__file__)).parent.parent.parent
os.chdir(ROOT_DIR)
# -------------------------------------------------------------------------------------------------------------------- #

import os
from docs.automacdoc.automacdoc import write_doc

if __name__ == "__main__":

    # Root directory of the project.
    root_dir = ''

    # The directory to document. (Creates md readme files from all python files found in document_dir).
    document_dir = os.path.join(root_dir, 'lib')

    # Directories within document_dir that you do not wanted documented.
    skip_dirs = []

    # Create md files.
    try:
        write_doc(document_dir, root_dir, skip_dirs)
    except Exception as error:
        print("[-] Error ", str(error))


