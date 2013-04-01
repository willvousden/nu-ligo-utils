def tail(f, n):
    """Returns the last ``n`` lines of the file ``f``."""

    if not (isinstance(f, file)):
        with open(f, 'r') as inp:
            return tail(f, n)

    i = 0
    lines = []
    for l in f:
        if i < n:
            lines.append(l)
            i = i+1
        else:
            lines[i%n] = l
            i = i+1

    if i < n:
        raise ValueError('tail: file must have at least {0:d} lines'.format(n))

    return lines[i%n:] + lines[:i%n]
