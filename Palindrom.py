# Function to check if a number is a palindrome
def is_palindrome(num):
    # Convert the number to a string
    num_str = str(num)
    
    # Reverse the string and compare with the original
    if num_str == num_str[::-1]:
        return True
    else:
        return False

# Input from the user
number = int(input("Enter a number: "))

# Check if the number is a palindrome
if is_palindrome(number):
    print(f"{number} is a palindrome")
else:
    print(f"{number} is not a palindrome")
