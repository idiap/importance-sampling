#!/usr/bin/env python
#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import argparse
from itertools import chain, ifilter
import os
from os import path
from subprocess import PIPE, Popen


class Header(object):
    """Represents the copyright header for a source file"""
    COPY = """#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by """

    def __init__(self, start=-1, stop=-1, content=None):
        self.start = start
        self.stop = stop
        self.content = content

    def update(self, filepath, dry_run=False):
        def peek(f):
            pos = f.tell()
            c = f.read(1)
            f.seek(pos)

            return c

        new_content = Header.get_content_for_file(filepath)
        needs_update = new_content != self.content or self.start < 0

        if not needs_update:
            return

        # Do the dry run
        if dry_run:
            print new_content
            return

        # Open both files and do the copy while updating the header
        with open(filepath) as f_in, open(filepath+".header", "w") as f_out:
            # Copy the comments that appear on top
            while peek(f_in) == "#":
                f_out.write(f_in.readline())

            # Consume one new line
            if peek(f_in) in ["\r", "\n"]:
                f_in.readline()

            # Add the new header
            start = f_out.tell()
            f_out.write(new_content)

            # If the file had a header skip it while writing the rest of the data
            if self.start > 0:
                f_out.write(f_in.read(max(0, self.start - f_in.tell())))
                f_in.seek(max(f_in.tell(), self.stop))
            f_out.write(f_in.read())

        stat = os.stat(filepath)
        os.chmod(filepath+".header", stat.st_mode)
        os.rename(filepath+".header", filepath)


    @classmethod
    def from_file(cls, filepath):
        # Create an empty object to be filled with contents
        header = cls()

        # Read the file contents into memory
        with open(filepath) as f:
            contents = f.read()

        # Find the copyright disclaimer
        start = contents.find("#\n# Copyright")
        if start < 0:
            return header
        end = contents.find("\n#\n\n", start) + 4

        # Fill in the header
        header.start = start
        header.end = end
        header.content = contents[start:end]

        return header

    @staticmethod
    def get_content_for_file(filepath):
        """Return the generated header for the file"""
        # Call into git to get the list of authors
        p = Popen(["git", "shortlog", "-se", "--", filepath], stdout=PIPE)
        out, _ = p.communicate()
        authors = [
            l.split("\t")[1].strip()
            for l in out.splitlines()
            if l != ""
        ]

        return Header.COPY + ",\n# ".join(authors) + "\n#\n\n"


def is_python_file(path):
    return path.endswith(".py")


def in_directory(directory):
    if directory[0] == path.sep:
        directory = directory[1:]
    if directory[-1] == path.sep:
        directory = directory[:-1]
    def inner(x):
        return path.sep + directory + path.sep in x
    return inner


def _all(*predicates):
    def inner(x):
        return all(p(x) for p in predicates)
    return inner


def _not(predicate):
    def inner(x):
        return not predicate(x)
    return inner


def walk_directories(root):
    """'find' in a generator function."""
    for child in os.listdir(root):
        if child.startswith("."):
            continue

        full_path = path.join(root, child)
        if path.isfile(full_path):
            yield full_path
        elif full_path.endswith((path.sep+".", path.sep+"..")):
            continue
        elif path.islink(full_path):
            continue
        else:
            for fp in walk_directories(full_path):
                yield fp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate file copywrite headers and prepend them to "
                     "the files in the repository")
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't actually change anything just write the headers to STDOUT"
    )
    parser.add_argument(
        "--blacklist",
        type=lambda x: x.split(":"),
        default=[],
        help="A colon separated list of directories to blacklist"
    )

    args = parser.parse_args()

    # Loop over all python files
    predicate = _all(
        is_python_file,
        _all(*map(_not, map(in_directory, args.blacklist)))
    )
    for source_file in ifilter(predicate, walk_directories(".")):
        print source_file
        header = Header.from_file(source_file)
        header.update(source_file, args.dry_run)
