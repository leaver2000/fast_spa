```
# tables.pxd
cdef int get_tab()

# tables.pyx
cdef int get_tab():
    return 1


# lib.pyx
from tables import get_tab

cdef _f():
    print(get_tab())

def f():
    _f()

python -c 'import fastspa.lib as l; print(type(l.get_tab()))'
```