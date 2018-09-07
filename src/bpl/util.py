import os

from tempfile import mkstemp


class ModelNotFitError(Exception):
    pass


class ModelNotConvergedWarning(Warning):
    pass


class suppress_output:
    """ Suppress stdout and stderr from stan. """

    def __init__(self):
        self.null_fds = [
            mkstemp(),
            mkstemp()
        ]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0][0], 1)
        os.dup2(self.null_fds[1][0], 2)

        return self

    @property
    def stdout(self):
        with open(self.null_fds[0][1], "r") as fp:
            contents = fp.read()
        return contents

    @property
    def stderr(self):
        with open(self.null_fds[1][1], "r") as fp:
            contents = fp.read()
        return contents

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        for fd in self.save_fds:
            os.close(fd)

        for fd, p in self.null_fds:
            os.close(fd)
            os.unlink(p)