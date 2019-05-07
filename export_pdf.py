import os
import re
import shutil
import sys


class TempDirContext:
    """Context manager for working in a temporary folder"""
    def __init__(self, filepath):
        self.tmp_dir = '/tmp/spam/'
        self.filepath = filepath
        self.working_dir = os.getcwd()

    def __enter__(self):
        shutil.rmtree(self.tmp_dir, True)
        os.mkdir(self.tmp_dir)
        shutil.copy2(self.filepath, "/".join([self.tmp_dir, os.path.basename(self.filepath)]))
        os.chdir(self.tmp_dir)
        return self

    def __exit__(self, *args):
        os.chdir(self.working_dir)
        shutil.rmtree(self.tmp_dir)




nb_path = sys.argv[1] if len(sys.argv) > 1 else input('enter notebook path\n')

# regular expressions
color_begin_expr = re.compile("<font color=.*?>")
color_end_expr = re.compile('</font>')
newline_expr = re.compile('\"\\n\"\n,')
title_expr = re.compile('\\\\subsection{(?P<title>.*?)}')
old_title_expr = re.compile('\\\\title{.*\}')
title_autor_string = "\\\\title{{{}}}\n\\\\author{{Jonas Sitzmann}}\n"

content = None
with open(nb_path, 'r') as file:
    content = ''.join(file.readlines())

# manipulate the jupyter notebook
content = color_begin_expr.sub(r'\\\\textcolor{blue}{', content)
content = color_end_expr.sub('}', content)
content = newline_expr.sub('', content)

tmp = 'tmp'
nb_tmp, tex_tmp, pdf_tmp = ['.'.join([tmp, ending]) for ending in ['ipynb', 'tex', 'pdf']]
with TempDirContext(nb_path) as context:
    with open(nb_tmp, 'w') as out_file:
        out_file.write(content)
    os.system('jupyter-nbconvert --to=latex {}'.format(nb_tmp))
    tex_content = None
    with open(tex_tmp, 'r') as file:
        tex_content = ''.join(file.readlines())
    title = title_expr.search(tex_content).group(1)

    # manipulate the latex code
    tex_content = old_title_expr.sub(title_autor_string.format(title), tex_content)
    tex_content = title_expr.sub('', tex_content)
    with open(tex_tmp, 'w') as out_file:
        out_file.write(tex_content)

    # create target pdf file
    os.system('pdflatex {}'.format(tex_tmp))
    new_pdf_path = '{}/{}.pdf'.format(context.working_dir, nb_path.split('.')[0])
    shutil.copy2(pdf_tmp, new_pdf_path)
