cdef class Shrubbery:
    def __init__(self, int w, int l):
        self.width = w
        self.length = l

cdef Shrubbery standard_shrubbery():
    return Shrubbery(3, 7)