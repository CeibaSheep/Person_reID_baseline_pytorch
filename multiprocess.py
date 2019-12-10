from multiprocessing import Process, Value, Array

# def f(n):
#     flag = True
#     while flag :
#         if n.value == 0 :
#             flag = False
#             print (1)
#     print (2)

# if __name__ == '__main__':
#     num = Value('i', 1)

#     p = Process(target=f, args=(num))
#     p.start()
#     # p.join()

#     print(num.value)
#     num.value = 0
#     print('end')

# from multiprocessing import Process, Value, Array

def f(n, a):
    # n.value = 3.1415927
    while (n.value) :
        print n.value
        # a[i] = -a[i]
    print 2

if __name__ == '__main__':
    num = Value('d', True)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    num.value = False
    print(arr[:])