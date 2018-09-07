import os

from tempfile import mkstemp


class ModelNotFitError(Exception):
    pass


class ModelNotConvergedWarning(Warning):
    pass


class suppress_output:
    """ Suppress all stdout and stderr. """

    def __init__(self):
        """
        self.null_fds = [
            os.open(os.devnull, os.O_RDWR),
            os.open(os.devnull, os.O_RDWR)
        ]
        """
        self.null_fds = [
            mkstemp(),
            mkstemp()
        ]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
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
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close the null files and descriptors.
        for fd in self.save_fds:
            os.close(fd)

        for fd, p in self.null_fds:
            os.close(fd)
            os.unlink(p)