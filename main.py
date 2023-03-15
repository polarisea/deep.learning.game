import multiprocessing as mp

from recognized import recognized
from game import game


if __name__ == '__main__':
    act = mp.Value('i')
    t2 = mp.Process( target=recognized, args=(act,) )
    t1 = mp.Process( target=game, args=(act,) )
    t2.start()
    t1.start()
    t2.join()
    t1.join()
