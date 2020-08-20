import torch
import cl.torch as cl


def main():
    clayers = cl.ConformalLayers()
    clayers.enqueue_modules(
        cl.Conv1d(1, 1, 5),
        cl.SRePro(),
        cl.Dropout(0.5),
        cl.AvgPool1d(2),
        cl.Conv1d(1, 1, 3, bias=False),
        cl.SRePro(),
    )
    
    print(clayers)

if __name__ == "__main__":
    main()
