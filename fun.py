from cl.pytorch import ConformalLayers, AveragePooling, Dropout


def main():
    clayers = ConformalLayers()
    clayers.enqueue_module(Dropout(3))
    clayers.enqueue_module(AveragePooling(2))
    clayers.enqueue_module(Dropout(4))
    print(clayers)


if __name__ == "__main__":
    main()
