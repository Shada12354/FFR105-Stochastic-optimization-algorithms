import math
# ToDo: Uncomment the line below ("import ...") once you have implemented plot_iterations.
# You may need to install Matplotlib first: pip install matplotlib
import matplotlib.pyplot as plt

# ============================================
# get_polynomial_value:
# ============================================

# ToDo: Write the get_polynomial_value function
# Uncomment the line below and implement the function:
def get_polynomial_value(x, polynomial_coefficients):
    #polynomial_value = a0 as intial value, then adding a1*x, a2*x^2,...
    polynomial_value = polynomial_coefficients[0]

    for i in range(1, len(polynomial_coefficients)):
        polynomial_value += polynomial_coefficients[i] * (x**i)
    return polynomial_value

# ============================================
# differentiate_polynomial:
# ============================================

# ToDo: Write the differentiate_polynomial function

# Uncomment the line below and implement the function:
def differentiate_polynomial(polynomial_coefficients, derivative_order):

    if derivative_order >= len(polynomial_coefficients):
        derivative_coefficients = []
        return derivative_coefficients

    derivative_coefficients = polynomial_coefficients[:]

   #Starting at index 1, x^0 never gives a derivative
    for m in range(derivative_order):
        derivative_coefficients = [n * derivative_coefficients[n] for n in range(1,len(derivative_coefficients))]

    return derivative_coefficients


# ============================================
# step_newton_raphson:
# ============================================

# ToDo: Write the step_newton_raphson function

# Uncomment the line below and implement the function:
def step_newton_raphson(x, f_prime, f_double_prime):
    #x_(j+1) = x_next
    #x_j = x
    x_next = x - (f_prime / f_double_prime)
    return x_next

# ============================================
# run_newton-raphson:
# ============================================
     
# ToDo: Write the run_newton-raphson function

# Uncomment the line below and implement the function:
def run_newton_raphson(polynomial_coefficients, starting_point, tolerance, maximum_number_of_iterations):
    #If length of polynomial coefficients < 3:
    # highetst power is x^1 -> no second derivative
    if len(polynomial_coefficients) < 3:
        print("Error, second derivative of the polynomial must be greater than zero.")
        return

    X = [starting_point]
    F = [get_polynomial_value(starting_point,polynomial_coefficients)]

    for j in range(1,maximum_number_of_iterations):
        f_prime = get_polynomial_value(X[j-1],differentiate_polynomial(polynomial_coefficients,1))
        f_bizz = get_polynomial_value(X[j-1],differentiate_polynomial(polynomial_coefficients,2))
        # x(j +1)
        x_next = step_newton_raphson(X[j-1], f_prime, f_bizz)
        X.append(x_next)
        F.append(get_polynomial_value(x_next, polynomial_coefficients))
        result = [X, F]

        if abs(X[j] - X[j-1]) < tolerance:
            break

    return [X, F]

# ============================================
# plot_iterations:
# ============================================

# ToDo: Write the plot_iterations function
#
# Here, you should use matplotlib. 
# Note: You must uncomment the second "import ..." statement above
# Then uncomment the line below and implement the function:
def plot_iterations(polynomial_coefficients,iterations):
    X,F = iterations
    #Newton-Raphson points
    plt.plot(X, F, 'o', color= 'blue')

    #whole curve
    x_min = min(X)-1
    x_max = max(X)+1
    x_axis = [x_min + i * (x_max - x_min) / 300 for i in range(301)]
    y_axis = [get_polynomial_value(x,polynomial_coefficients) for x in (x_axis)]
    plt.plot(x_axis, y_axis)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Newton-raphson's method")
    plt.show()

# ============================================
# Main loop
# ============================================

tolerance = 0.00001
maximum_number_of_iterations = 10

polynomial_coefficients = [10,-2,-1,1]
starting_point = 2

# ToDo: Uncomment the two lines below, once you have implemented the functions above:
iterations = run_newton_raphson(polynomial_coefficients, starting_point, tolerance, maximum_number_of_iterations)
plot_iterations(polynomial_coefficients, iterations)
