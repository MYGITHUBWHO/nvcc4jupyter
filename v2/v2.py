import os
import subprocess

from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython.config import Configurable
from IPython.utils.traitlets import Unicode, Bool
from common import helper

compiler = '/usr/local/cuda/bin/nvcc'

@magics_class
class NVCCPluginV2(Magics):

    static_dir   = Bool(False, config=True, help='Use static output directory')
    relative_dir = Unicode(u'.', config=True, help='the relative_dir of source codes from current dir')
    out          = Unicode(u'result.out', config=True, help='filename of output binary')

    def __init__(self, shell):
        super(NVCCPluginV2, self).__init__(shell)
        self.argparser = helper.get_argparser()

        if self.static_dir:
            current_dir = os.getcwd()
            self.output_dir = os.path.join(current_dir, self.relative_dir)
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
                print(f'created output directory {self.output_dir}')

            print(f'Set up cuda environment at {self.output_dir}')
        else:
            self.output_dir = '.'

        print(f'Default out bin {self.out}')

    @staticmethod
    def compile(output_dir, file_paths, out):
        res = subprocess.check_output([compiler, '-I' + output_dir, file_paths, "-o", out], stderr=subprocess.STDOUT)
        print(res)

    def run(self, timeit=False):
        if timeit:
            stmt = f"subprocess.check_output(['{self.out}'], stderr=subprocess.STDOUT)"
            output = self.shell.run_cell_magic(magic_name="timeit", line="-q -o import subprocess", cell=stmt)
        else:
            output = subprocess.check_output([self.out], stderr=subprocess.STDOUT)
            output = output.decode('utf8')

        return output

    @magic_arguments()
    @argument('-n', '--name', type=str, help='file name that will be produced by the cell. must end with .cu extension')
    @argument('-c', '--compile', type=bool, default=False, help='Should be compiled?')
    @argument('-t', '--timeit', type=bool, default=False, help='Time the code?')
    @cell_magic
    def cuda(self, line='', cell=None):
        args = parse_argstring(self.cuda, line)
        ex = args.name.split('.')[-1]
        if ex not in ['cu', 'h']:
            raise Exception('name must end with .cu or .h')

        if not os.path.exists(self.output_dir):
            print(f'Output directory does not exist, creating')
            try:
                os.mkdir(self.output_dir)
            except OSError:
                print(f"Creation of the directory {self.output_dir} failed")
            else:
                print(f"Successfully created the directory {self.output_dir}")

        file_path = os.path.join(self.output_dir, args.name)
        with open(file_path, "w") as f:
            f.write(cell)

        if args.compile:
            try:
                self.compile(self.output_dir, file_path, self.out)
                output = self.run(timeit=args.timeit)
            except subprocess.CalledProcessError as e:
                print(e.output.decode("utf8"))
                output = None
        else:
            output = f'File written in {file_path}'

        return output

    @cell_magic
    def cuda_run(self, line='', cell=None):
        try:
            args = self.argparser.parse_args(line.split())
        except SystemExit:
            self.argparser.print_help()
            return

        try:
            cuda_src = os.listdir(self.output_dir)
            cuda_src = [os.path.join(self.output_dir, x) for x in cuda_src if x[-3:] == '.cu']
            print(f'found sources: {cuda_src}')
            self.compile(self.output_dir, ' '.join(cuda_src), self.out)
            output = self.run(timeit=args.timeit)
        except subprocess.CalledProcessError as e:
            print(e.output.decode("utf8"))
            output = None

        return output
