from dataclasses import dataclass, astuple
from scipy.optimize import basinhopping
import numpy as np

@dataclass
class Model:
    param1: float
    param2: float
    # Add more parameters if needed

    def to_array(self):
        return astuple(self)

def model_eval(model: Model):
    # Replace this with your actual model evaluation logic
    return abs(model.param1 - 2) + abs(model.param2**2 - 2)

def basinhopping_wrapper(x):
    model = Model(*x)
    return model_eval(model)


def main():
    # Define the initial model
    initial_model = Model(1, 1)  # Update with your actual initial guess

    # Convert the initial model to an array for basinhopping
    initial_guess = initial_model.to_array()

    # Minimization using basinhopping
    result = basinhopping(basinhopping_wrapper, initial_guess, niter=100)

    # Extract the best fit model
    best_model = Model(*result.x)

    print("Best model found:", best_model)
    print("Function value:", result.fun)

if __name__ == '__main__':
    main()