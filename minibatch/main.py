from generator import MiniBatchGenerator

def main():
    generator = MiniBatchGenerator(10, x_mapper, y_mapper)
    generator.split_train_test()

    for i in range(10):
        print('load train mini-batch')
        generator.reset()
        while True:
            X, y = generator.load_next_train_batch(100)
            if X is None:
                break
            print(X)

        print('load test mini-batch')
        while True:
            X, y = generator.load_next_test_batch(100)
            if X is None:
                break
            print(X)

def x_mapper(item):
    return "X" + str(item)

def y_mapper(item):
    return "y" + str(item)

if __name__ == '__main__':
    main()
