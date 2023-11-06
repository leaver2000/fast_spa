from .shrubbing cimport Shrubbery, standard_shrubbery

def main():
    cdef Shrubbery sh
    sh = standard_shrubbery()
    print("Shrubbery size is", sh.width, 'x', sh.length)