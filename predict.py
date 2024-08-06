"""
This program predicts the price of a car with its kilometrage.
"""

from functions import predict, get_weights

def main() -> None:
    """Executes the program."""
    while True:
        try:
            kilometrage = float(input("Please enter Kilometrage: "))
            if kilometrage < 0:
                print("Error: Negative mileage. Please enter a non-negative value.")
                continue
            break
        except ValueError:
            print("Error: Cannot cast input to float. Please enter a valid number.")
    
    theta0, theta1 = get_weights()
    prediction = predict(theta0, theta1, kilometrage)
    
    rounded_prediction = round(prediction)
    
    print("Estimated price for {} kms: {} €".format(int(kilometrage), rounded_prediction))
    
    if theta0 == 0 and theta1 == 0:
        print("Note: it seems that the model is not trained yet. Run train.py to set weights.")
        return 

    response = input("Is it what you expected? (yes/no): ").lower()
    
    if response == 'yes':
        print("Great! I'll buy it!")
    elif response == 'no':
        print("Sorry to hear that. I'll have to try to adjust my algorithm a little more.")

if __name__ == "__main__":
    main()