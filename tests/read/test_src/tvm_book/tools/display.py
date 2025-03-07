from dataclasses import dataclass
from pathlib import Path
import textwrap

@dataclass
class Tree:
    prefix: str = "|=> "
    indent: int = 4 # 缩进大小
    def __post_init__(self):
        self.trees = []

    def generate_tree(self, file, n=0):
        if file.is_file():
            self.trees.append(textwrap.indent(file.name, prefix=" "*n+self.prefix))
        elif file.is_dir():
            self.trees.append(textwrap.indent(str(file.relative_to(file.parent)), prefix=" "*n+self.prefix))
            for cp in file.iterdir():
                self.generate_tree(cp, self.indent + n)

    def __call__(self, file, n=0):
        file = Path(file)
        self.generate_tree(file, n)
        return '\n'.join(self.trees)
