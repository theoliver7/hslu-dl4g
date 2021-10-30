import numpy as np


class HandSampler:
    available_cards = np.ones(34,int)

    def main(self):
        print(self.available_cards)


if __name__ == '__main__':
    HandSampler().main()