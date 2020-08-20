import torch
import cl.torch as cl


def main():
    clayers = cl.ConformalLayers()
    clayers.enqueue_modules(
        cl.Conv1d(1, 1, 5),
        cl.Dropout(0.5),
        cl.Conv1d(1, 1, 3),
        cl.SRePro(),
        cl.AvgPool1d(2)
    )
    print(clayers)
    print()


if __name__ == "__main__":
    main()
